"""Integration tests for the evaluate() pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from transformers import T5Tokenizer

from model.model import T5EmbeddingModel
from eval.evaluate import evaluate, EvalResult


BASE_MODEL = "t5-small"


@pytest.fixture(scope="module")
def model_tok():
    tok = T5Tokenizer.from_pretrained(BASE_MODEL)
    mdl = T5EmbeddingModel(base_model=BASE_MODEL)
    return mdl, tok


# Minimal eval data: correct doc is in corpus, others are distractors
EVAL_DATA = [
    {
        "query": "capital of France",
        "positives": ["Paris is the capital of France."],
        "corpus": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Madrid is the capital of Spain.",
            "Rome is the capital of Italy.",
        ],
    },
    {
        "query": "Python programming language",
        "positives": ["Python is a high-level programming language."],
        "corpus": [
            "Python is a high-level programming language.",
            "Java is a compiled language.",
            "The Amazon river is the longest in the world.",
            "Photosynthesis converts sunlight to energy.",
        ],
    },
]


class TestEvaluatePipeline:
    def test_returns_eval_result(self, model_tok):
        model, tok = model_tok
        result = evaluate(model, tok, EVAL_DATA)
        assert isinstance(result, EvalResult)

    def test_n_queries(self, model_tok):
        model, tok = model_tok
        result = evaluate(model, tok, EVAL_DATA)
        assert result.n_queries == 2

    def test_metrics_in_range(self, model_tok):
        model, tok = model_tok
        result = evaluate(model, tok, EVAL_DATA)
        for val in [result.mrr_10, result.recall_1, result.recall_5,
                    result.recall_10, result.ndcg_10, result.map_10]:
            assert 0.0 <= val <= 1.0

    def test_str_output(self, model_tok):
        model, tok = model_tok
        result = evaluate(model, tok, EVAL_DATA)
        s = str(result)
        assert "MRR" in s and "NDCG" in s and "R@1" in s
