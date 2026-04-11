import logging
import time

import chainlit as cl
import chromadb

from src.config import ConfigError, load_config
from src.generation.answerer import answer, build_source_elements
from src.generation.condenser import Condenser
from src.health_check import check_models, check_ollama
from src.ingestion.ingest import IngestResult, ingest_folder
from src.ingestion.scanner import scan_folder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import make_ollama_embed_fn
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _build_retriever(store: VectorStore, config) -> HybridRetriever:
    """Build BM25 index and hybrid retriever from current store contents."""
    bm25_index = BM25Index()
    doc_count = store.count()
    if doc_count > 0:
        start = time.time()
        texts, metadatas = store.get_all_texts_and_metadatas()
        bm25_index.build(texts, metadatas)
        elapsed = time.time() - start
        logger.info("BM25 index built from %d chunks in %.2fs", doc_count, elapsed)

    return HybridRetriever(
        vector_store=store,
        bm25_index=bm25_index,
        config=config.retrieval,
    )


async def _run_ingestion(store: VectorStore, config, **kwargs) -> int:
    """Ingest documents from configured folder. Returns chunk count."""
    folder = config.paths.documents
    if str(folder) and folder.exists():
        result = ingest_folder(
            folder,
            store,
            recursive=config.scanning.recursive,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            **kwargs,
        )
        if result.ingested > 0:
            logger.info(
                "Ingested %d new files, skipped %d, failed %d",
                result.ingested, result.skipped, result.failed,
            )
        elif result.skipped > 0:
            logger.info("All %d files unchanged, skipped", result.skipped)

    return store.count()


@cl.on_chat_start
async def on_chat_start():
    try:
        config = load_config()
    except ConfigError as e:
        await cl.Message(content=f"**Configuration Error**\n\n{e}").send()
        return

    cl.user_session.set("config", config)

    ollama_result = await check_ollama()
    if not ollama_result.ok:
        await cl.Message(
            content=f"**Health Check Failed**\n\n{ollama_result.message}"
        ).send()
        return

    required_models = [config.models.llm, config.models.embedding]
    models_result = await check_models(required_models)
    if not models_result.ok:
        await cl.Message(
            content=f"**Health Check Failed**\n\n{models_result.message}"
        ).send()
        return

    # Set up vector store with persistent ChromaDB and Ollama embeddings
    embed_fn = make_ollama_embed_fn(model=config.models.embedding)
    chroma_client = chromadb.PersistentClient(
        path=str(config.paths.chroma_db),
    )
    store = VectorStore(embed_fn=embed_fn, client=chroma_client)
    cl.user_session.set("store", store)

    # Auto-scan and ingest new/changed documents on startup
    doc_count = await _run_ingestion(store, config)

    # Build retriever
    retriever = _build_retriever(store, config)
    cl.user_session.set("retriever", retriever)

    # Load cross-encoder reranker
    reranker = Reranker(model_name=config.retrieval.reranker_model)
    cl.user_session.set("reranker", reranker)

    # Create condenser for follow-up questions and init chat history
    condenser = Condenser(model=config.models.llm)
    cl.user_session.set("condenser", condenser)
    cl.user_session.set("chat_history", [])

    # Set up Chainlit settings panel
    settings = await cl.ChatSettings(
        [
            cl.input_widget.TextInput(
                id="documents_folder",
                label="Documents Folder",
                initial=str(config.paths.documents),
            ),
            cl.input_widget.Switch(
                id="recursive_scan",
                label="Recursive Scanning",
                initial=config.scanning.recursive,
            ),
        ]
    ).send()

    # Add re-ingest action button
    actions = [
        cl.Action(
            name="reingest",
            payload={"value": "reingest"},
            label="Re-ingest Documents",
            description="Force re-ingest all documents from the configured folder",
        )
    ]

    if doc_count > 0:
        await cl.Message(
            content=f"**Ready** \u2014 {doc_count} chunks indexed. Ask me anything!",
            actions=actions,
        ).send()
    else:
        await cl.Message(
            content="No documents indexed. Configure your documents folder in the settings panel (gear icon) or `config.yaml`.",
            actions=actions,
        ).send()


def _build_ingest_summary(result: IngestResult, total_files: int) -> str:
    """Build a human-readable ingestion summary."""
    parts = [f"**Ingestion complete.** {result.ingested}/{total_files} documents ingested."]

    if result.skipped > 0:
        parts.append(f"{result.skipped} unchanged (skipped).")

    if result.failed > 0:
        parts.append(f"{result.failed} failed.")
        parts.append("\n<details><summary>Failed documents</summary>\n")
        for filename, error in result.failures:
            parts.append(f"- **{filename}**: {error}")
        parts.append("\n</details>")

    return " ".join(parts)


@cl.action_callback("reingest")
async def on_reingest(action: cl.Action):
    """Handle manual re-ingest action button."""
    config = cl.user_session.get("config")
    store = cl.user_session.get("store")
    folder = config.paths.documents

    if not str(folder) or not folder.exists():
        await cl.Message(
            content="No documents folder configured. Set it in the settings panel first."
        ).send()
        return

    # Show document count before starting
    files = scan_folder(folder, recursive=config.scanning.recursive)
    total_files = len(files)
    await cl.Message(
        content=f"Re-ingesting {total_files} documents from `{folder}`..."
    ).send()

    # Run ingestion with force=True and per-file progress steps
    async with cl.Step(name="Re-ingestion Progress", type="tool") as step:
        result = ingest_folder(
            folder,
            store,
            recursive=config.scanning.recursive,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            force=True,
            on_progress=lambda name, status: logger.info("  %s: %s", name, status),
        )
        step.output = _build_ingest_summary(result, total_files)

    # Rebuild retriever with updated store
    retriever = _build_retriever(store, config)
    cl.user_session.set("retriever", retriever)

    summary = _build_ingest_summary(result, total_files)
    doc_count = store.count()
    await cl.Message(
        content=f"{summary}\n\n**Ready** \u2014 {doc_count} chunks indexed."
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Re-scan and ingest when settings change."""
    from pathlib import Path

    config = cl.user_session.get("config")
    store = cl.user_session.get("store")

    folder = Path(settings["documents_folder"]).expanduser()
    recursive = settings["recursive_scan"]

    if not folder.exists():
        await cl.Message(
            content=f"Folder not found: `{folder}`"
        ).send()
        return

    # Update config in session
    config.paths.documents = folder
    config.scanning.recursive = recursive

    await cl.Message(content=f"Scanning `{folder}`...").send()

    doc_count = await _run_ingestion(store, config)

    # Rebuild retriever with updated store
    retriever = _build_retriever(store, config)
    cl.user_session.set("retriever", retriever)

    await cl.Message(
        content=f"**Ready** \u2014 {doc_count} chunks indexed."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    retriever = cl.user_session.get("retriever")

    if retriever is None:
        await cl.Message(content="No retriever available. Please restart the app.").send()
        return

    store = cl.user_session.get("store")
    if store is None or store.count() == 0:
        await cl.Message(
            content="No documents indexed yet. Configure your documents folder in the settings panel (gear icon) or `config.yaml`."
        ).send()
        return

    # Condense follow-up questions into standalone queries
    condenser = cl.user_session.get("condenser")
    chat_history = cl.user_session.get("chat_history")
    query = condenser.condense(message.content, chat_history)

    # Retrieval pipeline with collapsible step display
    async with cl.Step(name="Retrieval", type="tool") as retrieval_step:
        # Step 1: Hybrid search
        async with cl.Step(name="Searching", type="tool") as search_step:
            candidates = retriever.retrieve(query)
            search_step.output = f"Found {len(candidates)} candidates via hybrid search (BM25 + semantic)"

        # Step 2: Reranking
        async with cl.Step(name="Reranking", type="tool") as rerank_step:
            reranker = cl.user_session.get("reranker")
            results = reranker.rerank(
                query, candidates, top_k=config.retrieval.rerank_top_k
            )
            rerank_step.output = f"Reranked to top {len(results)} by cross-encoder relevance"

        retrieval_step.output = f"Retrieved {len(results)} relevant chunks"

    # Stream the answer token by token
    msg = cl.Message(content="")
    async for token in answer(
        query,
        results,
        model=config.models.llm,
    ):
        await msg.stream_token(token)
    await msg.send()

    # Attach expandable source chunks below the answer
    for el_data in build_source_elements(results):
        element = cl.Text(
            name=el_data["name"],
            content=el_data["content"],
            display=el_data["display"],
        )
        await element.send(for_id=msg.id)

    # Update chat history (kept separate from answer context window)
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": msg.content})
