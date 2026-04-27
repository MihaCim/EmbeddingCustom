# EmbeddingCustom

Fine-tuned T5 embedding model — encoder-only, trained with Multiple Negatives Ranking Loss (MNRL).

## Structure

```
EmbeddingCustom/
├── model/
│   ├── model.py      # T5EmbeddingModel — encoder, pooling (mean/cls), L2 norm
│   ├── train.py      # Fine-tuning loop — PairDataset, MNRL loss, AdamW + warmup
│   └── infer.py      # Load checkpoint, encode sentences
├── eval/
│   ├── metrics.py    # MRR@k, Recall@k, NDCG@k, MAP@k
│   └── evaluate.py   # Full retrieval eval pipeline + CLI
└── tests/
    ├── test_model.py     # Shape, normalization, pooling, similarity
    ├── test_train.py     # Dataset, collate, loss properties
    ├── test_metrics.py   # Metric math correctness
    └── test_evaluate.py  # End-to-end evaluate() pipeline
```

## Quickstart

```bash
pip install -e ".[dev]"
```

### Train

```python
from model.train import train

pairs = [
    ("query text", "relevant document"),
    ...
]
train(pairs, base_model="t5-base", epochs=3, batch_size=32, save_path="outputs/checkpoint")
```

### Infer

```python
from model.infer import load_model, encode

model, tokenizer = load_model("outputs/checkpoint", base_model="t5-base")
embeddings = encode(["sentence one", "sentence two"], model, tokenizer)
# embeddings: torch.Tensor (N, hidden_dim), L2-normalized
```

### Evaluate

```bash
python eval/evaluate.py --data eval.jsonl --checkpoint outputs/checkpoint --base-model t5-base
```

Eval data format (JSONL):
```json
{"query": "...", "positives": ["..."], "corpus": ["...", "...", "..."]}
```


### Test

```bash
pytest
```

## Model

`T5EmbeddingModel` wraps `T5EncoderModel` (decoder discarded). Forward pass:

1. Encode tokens → last hidden states `(B, L, D)`
2. Pool → `(B, D)` via mean or CLS
3. L2 normalize (optional)

## Training

MNRL treats every other item in the batch as a negative. No explicit negative mining needed — larger batches = harder negatives.

Key hyperparameters: `lr=2e-5`, `batch_size=32`, `warmup_ratio=0.1`, `weight_decay=0.01`.
