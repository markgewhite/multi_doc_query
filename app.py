import logging
import time

import chainlit as cl
import chromadb

from src.config import ConfigError, load_config
from src.generation.answerer import answer, build_source_elements
from src.generation.condenser import Condenser
from src.health_check import check_models, check_ollama
from src.ingestion.ingest import ingest_folder
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


async def _run_ingestion(store: VectorStore, config) -> int:
    """Ingest documents from configured folder. Returns chunk count."""
    folder = config.paths.documents
    if str(folder) and folder.exists():
        ingested, skipped = ingest_folder(
            folder,
            store,
            recursive=config.scanning.recursive,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        if ingested > 0:
            logger.info("Ingested %d new files, skipped %d unchanged", ingested, skipped)
        elif skipped > 0:
            logger.info("All %d files unchanged, skipped", skipped)

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

    if doc_count > 0:
        await cl.Message(
            content=f"{doc_count} chunks indexed. Ask me anything!"
        ).send()
    else:
        await cl.Message(
            content="No documents indexed. Configure your documents folder in the settings panel (gear icon) or `config.yaml`."
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
        content=f"Done. {doc_count} chunks indexed."
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

    candidates = retriever.retrieve(query)

    # Rerank candidates with cross-encoder
    reranker = cl.user_session.get("reranker")
    results = reranker.rerank(
        query, candidates, top_k=config.retrieval.rerank_top_k
    )

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
