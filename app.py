import chainlit as cl
import chromadb

from src.config import ConfigError, load_config
from src.generation.answerer import answer, build_source_elements
from src.health_check import check_models, check_ollama
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_folder
from src.retrieval.embeddings import make_ollama_embed_fn
from src.retrieval.vector_store import VectorStore


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

    # Ingest documents if a folder is configured
    doc_count = store.count()
    if str(config.paths.documents) and config.paths.documents.exists():
        docs = load_folder(
            config.paths.documents,
            recursive=config.scanning.recursive,
        )
        if docs:
            chunks = chunk_documents(
                docs,
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap,
            )
            store.add_chunks(chunks)
            doc_count = store.count()
            await cl.Message(
                content=f"Ingested {len(docs)} pages into {doc_count} chunks. Ask me anything!"
            ).send()
            return

    if doc_count > 0:
        await cl.Message(
            content=f"{doc_count} chunks indexed. Ask me anything!"
        ).send()
    else:
        await cl.Message(
            content="No documents indexed. Configure your documents folder in `config.yaml` to get started."
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    store = cl.user_session.get("store")

    if store is None:
        await cl.Message(content="No document store available. Please restart the app.").send()
        return

    if store.count() == 0:
        await cl.Message(
            content="No documents indexed yet. Configure your documents folder in `config.yaml` first."
        ).send()
        return

    results = store.search(
        message.content,
        k=config.retrieval.semantic_top_k,
    )

    # Stream the answer token by token
    msg = cl.Message(content="")
    async for token in answer(
        message.content,
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
