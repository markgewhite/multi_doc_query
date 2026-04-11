"""Tests for the condenser module."""

from unittest.mock import MagicMock, patch

import pytest

from src.generation.condenser import Condenser


class TestCondenser:
    """Tests for the Condenser class."""

    @patch("src.generation.condenser.ollama")
    def test_condense_skips_when_no_history(self, mock_ollama):
        """First question (no history) returns the original question unchanged."""
        condenser = Condenser(model="llama3.1:8b")
        result = condenser.condense("What are filing deadlines?", [])

        assert result == "What are filing deadlines?"
        mock_ollama.chat.assert_not_called()

    @patch("src.generation.condenser.ollama")
    def test_condense_rewrites_with_history(self, mock_ollama):
        """Follow-up question is rewritten using chat history."""
        mock_ollama.chat.return_value = {
            "message": {"content": "What are the penalties for missing tax filing deadlines?"}
        }

        condenser = Condenser(model="llama3.1:8b")
        history = [
            {"role": "user", "content": "What are filing deadlines?"},
            {"role": "assistant", "content": "Filing deadlines are in April."},
        ]
        result = condenser.condense("And what about penalties?", history)

        assert result == "What are the penalties for missing tax filing deadlines?"
        mock_ollama.chat.assert_called_once()

    @patch("src.generation.condenser.ollama")
    def test_condense_passes_correct_model(self, mock_ollama):
        """Condenser uses the configured model for the LLM call."""
        mock_ollama.chat.return_value = {
            "message": {"content": "standalone question"}
        }

        condenser = Condenser(model="custom-model")
        condenser.condense("follow up", [{"role": "user", "content": "first"}])

        call_kwargs = mock_ollama.chat.call_args
        assert call_kwargs[1]["model"] == "custom-model"

    @patch("src.generation.condenser.ollama")
    def test_condense_includes_history_in_prompt(self, mock_ollama):
        """The condensation prompt includes chat history."""
        mock_ollama.chat.return_value = {
            "message": {"content": "standalone question"}
        }

        condenser = Condenser(model="llama3.1:8b")
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]
        condenser.condense("Tell me more", history)

        messages = mock_ollama.chat.call_args[1]["messages"]
        prompt_text = " ".join(m["content"] for m in messages)
        assert "What is Python?" in prompt_text
        assert "A programming language." in prompt_text
        assert "Tell me more" in prompt_text

    @patch("src.generation.condenser.ollama")
    def test_condense_strips_whitespace(self, mock_ollama):
        """Condensed output has leading/trailing whitespace stripped."""
        mock_ollama.chat.return_value = {
            "message": {"content": "  standalone question  \n"}
        }

        condenser = Condenser(model="llama3.1:8b")
        result = condenser.condense("follow up", [{"role": "user", "content": "first"}])

        assert result == "standalone question"
