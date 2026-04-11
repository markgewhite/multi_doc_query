"""Tests for cross-encoder reranker."""

from unittest.mock import MagicMock, patch

import pytest

from src.models import SearchResult
from src.retrieval.reranker import Reranker


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create search results where order should change after reranking."""
    return [
        SearchResult(
            text="The capital of France is Paris.",
            metadata={"filename": "geography.pdf", "page_number": 1},
            distance=0.3,
        ),
        SearchResult(
            text="Python is a programming language.",
            metadata={"filename": "tech.pdf", "page_number": 5},
            distance=0.4,
        ),
        SearchResult(
            text="Paris has many famous landmarks including the Eiffel Tower.",
            metadata={"filename": "travel.pdf", "page_number": 10},
            distance=0.5,
        ),
    ]


class TestReranker:
    """Tests for the Reranker class."""

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_returns_top_k_results(
        self, mock_cross_encoder_cls, sample_results
    ):
        """Reranker returns at most top_k results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.7]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        reranked = reranker.rerank("What is Paris?", sample_results, top_k=2)

        assert len(reranked) == 2

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_reorders_by_score(
        self, mock_cross_encoder_cls, sample_results
    ):
        """Reranker reorders results by cross-encoder score (highest first)."""
        mock_model = MagicMock()
        # Scores: first=0.9, second=0.1, third=0.7
        mock_model.predict.return_value = [0.9, 0.1, 0.7]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        reranked = reranker.rerank("What is Paris?", sample_results, top_k=3)

        assert reranked[0].text == "The capital of France is Paris."
        assert reranked[1].text == "Paris has many famous landmarks including the Eiffel Tower."
        assert reranked[2].text == "Python is a programming language."

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_updates_distance_to_score(
        self, mock_cross_encoder_cls, sample_results
    ):
        """Reranked results have cross-encoder scores as distance."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.7]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        reranked = reranker.rerank("What is Paris?", sample_results, top_k=3)

        assert reranked[0].distance == 0.9
        assert reranked[1].distance == 0.7
        assert reranked[2].distance == 0.1

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_passes_query_chunk_pairs(
        self, mock_cross_encoder_cls, sample_results
    ):
        """Reranker passes correct (query, text) pairs to the model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.5, 0.5]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        reranker.rerank("What is Paris?", sample_results, top_k=3)

        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[0] == ("What is Paris?", "The capital of France is Paris.")

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_input(self, mock_cross_encoder_cls):
        """Reranker handles empty input gracefully."""
        mock_model = MagicMock()
        mock_cross_encoder_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []
        mock_model.predict.assert_not_called()

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_top_k_larger_than_input(
        self, mock_cross_encoder_cls, sample_results
    ):
        """When top_k exceeds input size, return all results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.7]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        reranked = reranker.rerank("query", sample_results, top_k=10)

        assert len(reranked) == 3

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_model_loaded_with_correct_name(self, mock_cross_encoder_cls):
        """CrossEncoder is initialised with the configured model name."""
        mock_cross_encoder_cls.return_value = MagicMock()

        Reranker(model_name="bge-reranker-v2-m3")

        mock_cross_encoder_cls.assert_called_once_with("bge-reranker-v2-m3")
