# EmbeddingCustom

Fine-tuned T5 embedding model — encoder-only, trained with Multiple Negatives Ranking Loss.

## Features

- **T5 encoder-only** — decoder discarded, mean or CLS pooling, optional L2 normalization
- **MNRL training** — in-batch negatives, no explicit negative mining needed
- **Retrieval eval** — MRR@k, NDCG@k, Recall@k, MAP@k out of the box
- **Simple data format** — `(query, positive)` pairs for training, JSONL for eval

## Quick Start

```bash
pip install -e ".[dev]"
```

**Train**

```python
from model.train import train

pairs = [
    ("query text", "relevant document"),
    ...
]
train(pairs, base_model="t5-base", epochs=3, batch_size=32, save_path="outputs/checkpoint")
```

**Infer**

```python
from model.infer import load_model, encode

model, tokenizer = load_model("outputs/checkpoint", base_model="t5-base")
embeddings = encode(["sentence one", "sentence two"], model, tokenizer)
# shape: (N, hidden_dim), L2-normalized
```

**Evaluate**

```bash
python eval/evaluate.py --data eval.jsonl --checkpoint outputs/checkpoint --base-model t5-base
```

Eval data format (JSONL — one query per line):

```json
{"query": "...", "positives": ["..."], "corpus": ["...", "...", "..."]}
```

**Test**

```bash
pytest
```

## Why MNRL?

Standard contrastive loss needs explicit hard negatives — expensive to mine. MNRL treats every other item in the batch as a negative. Larger batch = harder negatives, no mining pipeline needed.

## Structure

```
model/
├── model.py      # T5EmbeddingModel
├── train.py      # Fine-tuning loop
└── infer.py      # Encode sentences from checkpoint
eval/
├── metrics.py    # MRR, Recall, NDCG, MAP
└── evaluate.py   # Retrieval eval pipeline + CLI
tests/
├── test_model.py
├── test_train.py
├── test_metrics.py
└── test_evaluate.py
```

## License

MIT
