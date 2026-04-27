"""Unit tests for retrieval metrics."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from eval.metrics import mrr, recall_at_k, ndcg_at_k, map_at_k


# perfect: correct doc always ranked first
PERFECT = [[1, 0, 0, 0, 0]] * 4

# worst: correct doc always last
WORST = [[0, 0, 0, 0, 1]] * 4

# mixed: hit at position 2
MID = [[0, 1, 0, 0, 0]] * 4


class TestMRR:
    def test_perfect(self):
        assert mrr(PERFECT, k=10) == pytest.approx(1.0)

    def test_worst_at_k5(self):
        # hit at pos 5, within k=10
        assert mrr(WORST, k=10) == pytest.approx(1 / 5)

    def test_no_hit_in_k(self):
        # hit at pos 5, but k=3 → miss
        assert mrr([[0, 0, 0, 0, 1]], k=3) == pytest.approx(0.0)

    def test_mid(self):
        assert mrr(MID, k=10) == pytest.approx(0.5)


class TestRecallAtK:
    def test_perfect_recall_1(self):
        assert recall_at_k(PERFECT, k=1) == pytest.approx(1.0)

    def test_worst_recall_1(self):
        assert recall_at_k(WORST, k=1) == pytest.approx(0.0)

    def test_worst_recall_5(self):
        assert recall_at_k(WORST, k=5) == pytest.approx(1.0)

    def test_multiple_positives(self):
        # 2 relevant, top-3 has 1 → recall = 0.5
        rel = [[1, 0, 0, 1, 0]]
        assert recall_at_k(rel, k=3) == pytest.approx(0.5)


class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k(PERFECT, k=10) == pytest.approx(1.0)

    def test_zero_when_no_rel(self):
        assert ndcg_at_k([[0, 0, 0]], k=10) == pytest.approx(0.0)

    def test_lower_for_later_hit(self):
        early = ndcg_at_k([[1, 0, 0, 0, 0]], k=5)
        late = ndcg_at_k([[0, 0, 0, 0, 1]], k=5)
        assert early > late


class TestMAP:
    def test_perfect(self):
        assert map_at_k(PERFECT, k=10) == pytest.approx(1.0)

    def test_no_hit(self):
        assert map_at_k([[0, 0, 0]], k=10) == pytest.approx(0.0)

    def test_two_positives(self):
        # hits at pos 1 and 3 → AP = (1/1 + 2/3) / 2 = 0.833
        rel = [[1, 0, 1, 0, 0]]
        assert map_at_k(rel, k=5) == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-4)
