"""Tests for data loading utilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from pathlib import Path

import pytest

from data.loader import load_jsonl, load_sample_pairs, combine


class TestLoadJsonl:
    def test_basic(self, tmp_path):
        p = tmp_path / "pairs.jsonl"
        p.write_text(
            '{"query": "q1", "positive": "p1"}\n'
            '{"query": "q2", "positive": "p2"}\n'
        )
        pairs = load_jsonl(p)
        assert pairs == [("q1", "p1"), ("q2", "p2")]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "pairs.jsonl"
        p.write_text('{"query": "q", "positive": "p"}\n\n')
        assert len(load_jsonl(p)) == 1

    def test_returns_list_of_tuples(self, tmp_path):
        p = tmp_path / "pairs.jsonl"
        p.write_text('{"query": "q", "positive": "p"}\n')
        result = load_jsonl(p)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)


class TestLoadSamplePairs:
    def test_nonempty(self):
        pairs = load_sample_pairs()
        assert len(pairs) > 0

    def test_all_tuples_of_strings(self):
        pairs = load_sample_pairs()
        for q, p in pairs:
            assert isinstance(q, str) and isinstance(p, str)
            assert q and p  # no empty strings

    def test_minimum_count(self):
        pairs = load_sample_pairs()
        assert len(pairs) >= 50


class TestCombine:
    def test_concatenates(self):
        a = [("q1", "p1")]
        b = [("q2", "p2"), ("q3", "p3")]
        assert combine(a, b) == [("q1", "p1"), ("q2", "p2"), ("q3", "p3")]

    def test_empty_inputs(self):
        assert combine([], []) == []
