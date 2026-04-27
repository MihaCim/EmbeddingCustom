"""Unit tests for training components (loss, dataset, collate)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from transformers import T5Tokenizer

from model.train import PairDataset, collate, mnrl_loss


BASE_MODEL = "t5-small"

PAIRS = [
    ("What is Python?", "Python is a programming language."),
    ("Capital of France?", "Paris is the capital of France."),
    ("How does TCP work?", "TCP provides reliable, ordered delivery of bytes."),
]


@pytest.fixture(scope="module")
def tokenizer():
    return T5Tokenizer.from_pretrained(BASE_MODEL)


class TestPairDataset:
    def test_len(self):
        ds = PairDataset(PAIRS)
        assert len(ds) == 3

    def test_getitem_returns_tuple(self):
        ds = PairDataset(PAIRS)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2


class TestCollate:
    def test_keys_present(self, tokenizer):
        batch = PAIRS[:2]
        enc_q, enc_p = collate(batch, tokenizer, max_length=32)
        assert "input_ids" in enc_q and "attention_mask" in enc_q
        assert "input_ids" in enc_p and "attention_mask" in enc_p

    def test_batch_size(self, tokenizer):
        batch = PAIRS[:2]
        enc_q, enc_p = collate(batch, tokenizer, max_length=32)
        assert enc_q["input_ids"].shape[0] == 2

    def test_truncation(self, tokenizer):
        batch = PAIRS
        enc_q, _ = collate(batch, tokenizer, max_length=8)
        assert enc_q["input_ids"].shape[1] <= 8


class TestMNRLLoss:
    def test_output_is_scalar(self):
        q = torch.randn(4, 64)
        p = torch.randn(4, 64)
        q = torch.nn.functional.normalize(q, dim=-1)
        p = torch.nn.functional.normalize(p, dim=-1)
        loss = mnrl_loss(q, p)
        assert loss.ndim == 0

    def test_loss_positive(self):
        q = torch.randn(4, 64)
        p = torch.randn(4, 64)
        loss = mnrl_loss(q, p)
        assert loss.item() > 0

    def test_perfect_alignment_low_loss(self):
        # q == p → diagonal scores dominate → low loss
        emb = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        loss = mnrl_loss(emb, emb, scale=20.0)
        assert loss.item() < 0.5

    def test_random_high_loss(self):
        # unrelated q, p → high loss
        q = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        p = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        loss = mnrl_loss(q, p, scale=20.0)
        assert loss.item() > 1.0
