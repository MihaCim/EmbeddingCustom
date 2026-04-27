"""Unit tests for T5EmbeddingModel."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from transformers import T5Tokenizer

from model.model import T5EmbeddingModel


BASE_MODEL = "t5-small"  # small for fast tests


@pytest.fixture(scope="module")
def tokenizer():
    return T5Tokenizer.from_pretrained(BASE_MODEL)


@pytest.fixture(scope="module")
def model():
    return T5EmbeddingModel(base_model=BASE_MODEL, pooling="mean", normalize=True)


def encode(texts, model, tokenizer):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        return model(**enc)


class TestOutputShape:
    def test_single(self, model, tokenizer):
        emb = encode(["hello"], model, tokenizer)
        assert emb.ndim == 2
        assert emb.shape[0] == 1

    def test_batch(self, model, tokenizer):
        emb = encode(["hello", "world", "foo"], model, tokenizer)
        assert emb.shape[0] == 3

    def test_hidden_dim_consistent(self, model, tokenizer):
        a = encode(["x"], model, tokenizer)
        b = encode(["x", "y"], model, tokenizer)
        assert a.shape[1] == b.shape[1]


class TestNormalization:
    def test_unit_norm(self, model, tokenizer):
        emb = encode(["test sentence"], model, tokenizer)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_no_normalize(self, tokenizer):
        m = T5EmbeddingModel(base_model=BASE_MODEL, normalize=False)
        emb = encode(["test"], m, tokenizer)
        norm = emb.norm(dim=-1).item()
        assert norm > 1.0 or norm != pytest.approx(1.0, abs=1e-5)


class TestPooling:
    def test_mean_pooling(self, tokenizer):
        m = T5EmbeddingModel(base_model=BASE_MODEL, pooling="mean", normalize=False)
        emb = encode(["mean pool test"], m, tokenizer)
        assert emb.shape[0] == 1

    def test_cls_pooling(self, tokenizer):
        m = T5EmbeddingModel(base_model=BASE_MODEL, pooling="cls", normalize=False)
        emb = encode(["cls pool test"], m, tokenizer)
        assert emb.shape[0] == 1

    def test_invalid_pooling(self, tokenizer):
        m = T5EmbeddingModel(base_model=BASE_MODEL, pooling="bad")
        enc = tokenizer(["x"], return_tensors="pt")
        with pytest.raises(ValueError):
            m(**enc)


class TestSimilarity:
    def test_identical_texts_high_sim(self, model, tokenizer):
        a = encode(["The quick brown fox"], model, tokenizer)
        b = encode(["The quick brown fox"], model, tokenizer)
        sim = (a * b).sum(dim=-1).item()
        assert sim > 0.99

    def test_different_texts_lower_sim(self, model, tokenizer):
        a = encode(["Paris is in France"], model, tokenizer)
        b = encode(["Python is a programming language"], model, tokenizer)
        sim = (a * b).sum(dim=-1).item()
        assert sim < 0.99
