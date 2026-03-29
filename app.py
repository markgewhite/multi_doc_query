import chainlit as cl

from src.config import ConfigError, load_config
from src.health_check import check_models, check_ollama


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

    await cl.Message(
        content="No documents indexed. Configure your documents folder in settings to get started."
    ).send()
