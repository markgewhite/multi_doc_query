"""Condense follow-up questions into standalone questions using chat history.

Not via LangChain: uses direct ollama.chat() for the condensation LLM call,
keeping full control over the prompt format and avoiding framework overhead.
"""

import ollama

CONDENSE_PROMPT = (
    "Given the following conversation history and a follow-up question, "
    "rewrite the follow-up question as a standalone question that captures "
    "the full context. Return ONLY the rewritten question, nothing else."
)


class Condenser:
    """Rewrites follow-up questions into standalone questions via LLM."""

    def __init__(self, model: str) -> None:
        self._model = model

    def condense(
        self,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        """Rewrite a follow-up question into a standalone question.

        If chat_history is empty, returns the question unchanged (no LLM call).
        """
        if not chat_history:
            return question

        history_text = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in chat_history
        )

        messages = [
            {"role": "system", "content": CONDENSE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Follow-up question: {question}"
                ),
            },
        ]

        response = ollama.chat(model=self._model, messages=messages)
        return response["message"]["content"].strip()
