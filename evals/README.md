# Hippocampus Retrieval Evals

Measures the repository's central claim: **for debugging agents, walking back
from a failure (temporally or causally) beats retrieving by similarity.**

## Methodology

### Dataset

`dataset.py` generates synthetic agent sessions, each containing:

- a **planted causal chain** (3–4 decisions linked by `parent_id`): a root
  cause (e.g. *"remove null check as an optimization"*) propagating to a
  failure event (e.g. *"production crash with NullPointerException"*)
- **distractors** that share the failure's vocabulary but are causally
  unrelated (an old NPE ticket, documentation about the same error, a similar
  crash in another service) — precisely the events similarity search loves
- **noise** (routine unrelated work)

Chain order is preserved; distractors and noise are interleaved at seeded
random positions. Generation is fully deterministic per seed.

### Task

Given the engineer's question about the failure (e.g. *"Why did the server
crash with a NullPointerException in token validation?"*), did retrieval
surface the **root cause** (and the rest of the chain)?

### Arms

| Arm | Uses | Information given |
|---|---|---|
| `semantic` | pgvector cosine only | query |
| `hybrid` | vector + keyword RRF | query |
| `temporal` | k events before the failure | query + ground-truth anchor offset |
| `causal` | causal-edge walk from the failure | query + ground-truth anchor offset |
| `anchor_causal_hybrid` | hybrid search finds the anchor, then causal walk | query only (full pipeline) |
| `anchor_causal_semantic` | vector search finds the anchor, then causal walk | query only (full pipeline) |

The `anchor_causal_*` arms are the honest end-to-end test of the README
thesis: no ground truth is provided, the system must find its own anchor.

**Results:** see [RESULTS.md](RESULTS.md) for measured numbers (MiniLM
embeddings, 40 sessions) including the distractor-density sweep and the bugs
the eval surfaced.

### Metrics

- `root_cause_recall@k` — fraction of cases where the true root cause is in the top k
- `chain_recall@k` — fraction of the full causal chain retrieved
- `mrr` — reciprocal rank of the root cause
- `ndcg@k` — graded gain (root cause weighted 2×, chain 1×)

### LLM judge (optional)

`--judge` feeds each arm's retrieved context to Claude and asks it to name the
root-cause decision ID (structured output). This measures whether the context
is *sufficient to reach the right conclusion*, not just whether the right row
was present. Default judge model is `claude-opus-4-8`; use
`--judge-model claude-haiku-4-5` for cheap bulk runs and spot-check
disagreements with the default model.

## Running

```bash
# Database with migrations applied (the test database works)
docker-compose -f docker-compose.test.yml up -d
export EVAL_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/hippocampus_test

# Real embeddings, no API key needed (sentence-transformers)
pip install -e ".[dev,local]"
python -m evals.run --provider local --sessions 40 --output results.json

# With the end-to-end judge
pip install -e ".[dev,local,evals]"
export ANTHROPIC_API_KEY=sk-ant-...
python -m evals.run --provider local --sessions 40 --judge
```

### Provider caveats

- `mock` — deterministic hash vectors with **no semantic signal**. Only the
  `temporal` and `causal` arms mean anything; CI uses this mode as a smoke
  test of the harness and the causal machinery.
- `local` — sentence-transformers (`all-MiniLM-L6-v2`); real semantics, free,
  reproducible. This is the default mode for reported results.
- `openai` — production embedding model; costs money, requires
  `OPENAI_API_KEY`.

## Interpreting results

The hypothesis predicts, with real embeddings:

1. `semantic` retrieves distractors (high lexical overlap with the failure)
   over the root cause (low overlap — it's about a null check, not a crash):
   low `root_cause_recall`.
2. `causal` is near-perfect *when causality was declared* — it follows edges,
   so distractors are structurally excluded.
3. `anchor_causal` approaches `causal` to the degree the anchor step finds
   the right failure event — the gap between them is the cost of anchor
   retrieval errors.
4. `temporal` sits in between, degrading as distractor density grows (the
   window fills with interleaved noise).

Negative results are results: if `semantic` matches `causal` on this dataset,
the distractors aren't hard enough or the thesis is wrong at this scale —
either finding goes in the write-up.
