# CLAUDE.md — EmbeddingCustom Development Guide

## Project Overview

**EmbeddingCustom** is a fine-tuned T5 embedding model for dense retrieval. The encoder-only architecture is trained with Multiple Negatives Ranking Loss (MNRL), which uses in-batch negatives — eliminating the need for an explicit negative mining pipeline.

**Stack:** Python 3.10+ · PyTorch · HuggingFace Transformers · pytest

---

## Repository Structure

```
EmbeddingCustom/
├── model/
│   ├── model.py        # T5EmbeddingModel — pooling, normalization
│   ├── train.py        # PairDataset, collate, mnrl_loss, train()
│   └── infer.py        # load_model(), encode()
├── eval/
│   ├── metrics.py      # mrr, recall_at_k, ndcg_at_k, map_at_k
│   └── evaluate.py     # evaluate(), evaluate_from_file(), CLI
├── data/
│   ├── loader.py       # load_jsonl, load_nli_pairs, load_msmarco_pairs
│   └── sample_pairs.jsonl
├── tests/
│   ├── test_model.py
│   ├── test_train.py
│   ├── test_metrics.py
│   ├── test_evaluate.py
│   └── test_data.py
└── pytest.ini
```

---

## Data Contracts

These are strict interface contracts. Changing them breaks downstream code.

### Training format

```python
Pairs = list[tuple[str, str]]  # (query, positive)
```

JSONL on disk — one pair per line:

```json
{"query": "...", "positive": "..."}
```

### Evaluation format

JSONL on disk — one query per line:

```json
{"query": "...", "positives": ["...", "..."], "corpus": ["...", "...", "..."]}
```

**Invariant:** every string in `positives` must also appear in `corpus`.

### `EvalResult` fields

`EvalResult` is a public interface. Field names must not change:

| Field | Type | Description |
|---|---|---|
| `mrr_10` | `float` | MRR@10 |
| `recall_1` | `float` | Recall@1 |
| `recall_5` | `float` | Recall@5 |
| `recall_10` | `float` | Recall@10 |
| `ndcg_10` | `float` | NDCG@10 |
| `map_10` | `float` | MAP@10 |
| `n_queries` | `int` | Number of evaluated queries |
| `extra` | `dict` | Optional extension point |

---

## Code Standards

### Python

- **Python 3.10+ syntax throughout.** Use built-in generics: `list[str]`, `tuple[str, str]`, `dict[str, int]`, `str | None`. Never import from `typing` for these.
- **Type hints on every function signature** — parameters and return types.
- **`pathlib.Path`** for all file paths. Never `os.path`.
- **f-strings only.** No `.format()`, no `%` formatting.
- **`dataclass`** for structured return types (see `EvalResult`).
- No dead code. Delete it — don't rename to `_unused_` or comment it out.
- No backwards-compatibility shims, re-exports, or feature flags.

### Comments

Write comments only when the **why** is non-obvious: a hidden constraint, a subtle invariant, a counterintuitive numerical choice, a workaround for a known bug. Never explain what the code does — well-named identifiers do that. Never reference the current task or PR in source code comments.

### Error handling

Validate only at system boundaries: CLI argument parsing, file I/O, external API calls. Trust internal guarantees from PyTorch and HuggingFace. Do not add defensive checks for states that cannot occur given the call site.

### Abstractions

Three similar lines of code is better than a premature helper function. Extract shared logic only when a third use case confirms the pattern. Do not design for hypothetical future requirements.

---

## ML & PyTorch Standards

### Device handling

Always pass `device` explicitly. Never rely on implicit device placement.

```python
# correct
enc = {k: v.to(device) for k, v in enc.items()}

# wrong — assumes CPU
enc = tokenizer(...)
```

### Inference

Every inference and evaluation function must be decorated with `@torch.no_grad()`. No exceptions.

```python
@torch.no_grad()
def encode(texts: list[str], model: T5EmbeddingModel, ...) -> torch.Tensor:
    ...
```

### Checkpoint loading

Always use `map_location` with `torch.load`. Omitting it breaks CPU-only environments.

```python
# correct
model.load_state_dict(torch.load(path, map_location="cpu"))

# wrong
model.load_state_dict(torch.load(path))
```

### Normalization

`T5EmbeddingModel` defaults to `normalize=True`. Embeddings are L2-normalized before any cosine similarity computation. Do not compute cosine similarity on unnormalized embeddings.

### MNRL loss

`mnrl_loss` uses `scale=20.0`. This constant was calibrated empirically — do not change it without benchmarking on a real retrieval dataset first.

```python
scores = scale * q_emb @ p_emb.T  # (B, B) — scale sharpens the softmax
```

### Gradient clipping

The training loop clips gradients at `max_norm=1.0`. This is required for stability with T5 and must not be removed.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### Pooling

`mean` pooling is the default and outperforms `cls` pooling on dense retrieval tasks. Do not change the default to `cls`.

---

## Testing

### Run tests

```bash
pytest            # all tests
pytest -k model   # single module
pytest -v         # verbose output (already in pytest.ini)
```

### Model used in tests

All tests use `t5-small` — same code paths as `t5-base`, runs in seconds. Never use `t5-base` in tests.

### Fixture scoping

Model and tokenizer fixtures are scoped to `module` to avoid re-loading HuggingFace weights on every test function.

```python
@pytest.fixture(scope="module")
def model():
    return T5EmbeddingModel(base_model="t5-small")
```

### What to test

| Layer | What to assert |
|---|---|
| Model | Output shape, dtype, L2 norm ≈ 1.0, identical inputs → identical outputs |
| Loss | Scalar output, positive value, perfect alignment → low loss |
| Metrics | Boundary conditions (perfect/worst/zero), mathematical identities |
| Data loaders | Return type contracts, blank line handling, empty input edge cases |
| Evaluate pipeline | `EvalResult` type, `n_queries`, metric range `[0.0, 1.0]` |

**Do not** mock the model forward pass. Tests run real inference on `t5-small`.

**Do not** test HuggingFace internals — tokenizer output shape, attention mask format, etc.

### Test-first for bug fixes

When fixing a bug: write the failing test first, confirm it fails, then fix the code.

---

## What Not To Change

The following are intentional design decisions. Do not alter them without a documented rationale and benchmark evidence.

| Item | Reason |
|---|---|
| `scale=20.0` in `mnrl_loss` | Calibrated for MNRL with T5-base embeddings |
| `max_norm=1.0` gradient clip | Required for stable T5 fine-tuning |
| `normalize=True` default | Cosine similarity requires L2-normalized vectors |
| `mean` pooling default | Outperforms CLS on dense retrieval |
| `EvalResult` field names | Public interface used by downstream consumers |
| In-batch negatives (MNRL) | Core design — avoids expensive negative mining |
| `t5-small` in tests | Speed; covers identical code paths |

---

## Git Conventions

### What to commit

- Source code, tests, `data/sample_pairs.jsonl`, `pytest.ini`

### Never commit

- Model checkpoints (`*.pt`)
- Large dataset files (NLI, MS MARCO downloads)
- `__pycache__/`, `.pyc` files (covered by `.gitignore`)
- Secrets, API keys, credentials

### Commit messages

Focus on **why**, not what. The diff shows what changed.

```
# good
Use mean pooling by default — outperforms CLS on BEIR benchmarks

# bad
Change pooling parameter default value from cls to mean
```
