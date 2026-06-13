# Eval Results

> Reproduce: `python -m evals.run --provider fastembed --sessions 40 --seed 42`
> (per-config JSON reports in [`evals/results/`](results/)).

## Setup

- **Dataset:** 40 synthetic sessions (4 incident scenarios × 10), each with a
  planted 3–4-step causal chain (root cause → failure) interleaved with
  distractors engineered to share the failure's vocabulary. Seed 42.
- **Embeddings:** `all-MiniLM-L6-v2` (ONNX via fastembed), zero-padded
  384 → 1536 to fit the schema column (cosine-invariant, rankings unaffected).
- **Database:** Postgres 16 + pgvector 0.6, Hippocampus migrations 001–002,
  with a schema-compatible stand-in for pg_kafka's tables (the eval exercises
  the retrieval SQL, not the broker; the queries are byte-identical).
- **Vector search is exhaustive** (`ivfflat.probes` = lists) — see finding 4.
- **Scope:** retrieval restricted to the incident's session (the realistic
  setting: you are debugging a specific run, and the distractors live in its
  log). The global-scope stress test is reported separately.
- The LLM-judge pass was not run (no API key in the eval environment); all
  numbers below are retrieval metrics.

## Headline (40 sessions, 6 distractors/session, k=5)

| arm | root_cause_recall@5 | chain_recall@5 | mrr | ndcg@5 |
|---|---|---|---|---|
| semantic (baseline RAG) | 0.425 | 0.696 | 0.104 | 0.518 |
| hybrid (RRF) | 0.425 | 0.696 | 0.104 | 0.518 |
| temporal window | 0.375 | 0.758 | 0.093 | 0.575 |
| **causal walk** (anchor given) | **1.000** | **1.000** | **0.313** | **0.840** |
| **pipeline** (search → walk, no ground truth) | **0.750** | 0.750 | 0.250 | 0.630 |

Similarity search finds the actual root cause in fewer than half the cases —
the top-5 fills with the failure's look-alikes. The causal walk is exact, and
the full pipeline (which must find its own anchor from the query) reaches
0.750.

## Distractor-density sweep (k=5, root_cause_recall@5)

| arm | d=2 | d=6 | d=12 |
|---|---|---|---|
| semantic | 1.000 | 0.425 | 0.100 |
| temporal | 0.850 | 0.375 | 0.100 |
| causal | 1.000 | 1.000 | 1.000 |
| pipeline | 1.000 | 0.750 | 0.750 |

This is the thesis in one table. With few distractors everything works.
As look-alike noise grows, similarity and temporal retrieval collapse
(the top-k and the window fill with distractors); the causal walk is
**structurally immune** — edges don't get noisier when the log does — and
the pipeline degrades only through its anchor-selection step, then holds.

## Other observations

- **k=10:** semantic/temporal saturate to 1.0 because sessions are small
  (~10–18 events; k=10 spans most of a session). At realistic scale, k must
  grow with noise for those arms; the causal walk needs k = chain length
  regardless. MRR still favors causal at k=10 (0.313 vs 0.180) — the cause
  ranks first instead of merely appearing.
- **Global scope** (query across all 40 sessions): every query-driven arm
  drops to ≈0 because the 4 scenario templates repeat across sessions and the
  query matches the wrong session's near-identical failure. Partly a dataset
  artifact, but it mirrors a real confound — a fleet of similar incidents —
  and the causal walk is again unaffected (1.000) once the anchor is known.

## What the eval caught (findings beyond the scoreboard)

1. **A real bug in `hybrid_search`** (fixed in this change): the topic filter
   was applied *after* RRF candidate selection, so scoped searches drew their
   candidate pools globally and then discarded out-of-scope rows — leaving
   near-arbitrary leftovers. Before the fix, hybrid's top-1 anchor landed on
   irrelevant noise 57% of the time and the pipeline scored 0.275; after,
   0.750. The candidate pool also gained a floor (`max(limit*4, 20)`) so
   `limit=1` anchor lookups don't starve.
2. **ivfflat built on an empty table loses data** (fixed in this change):
   first measured as recall@10 *below* recall@5 across runs — impossible for
   a fixed ranking — then reproduced at its worst: a freshly stored row
   completely invisible to `ORDER BY embedding <=> $1` because the index's
   centroids were trained on an empty table and the query probed the wrong
   list. The migration no longer creates an ANN index; exact scans are
   correct at this scale, and the documented graduation path is HNSW built
   after data load (HNSW has no training step, so no empty-table failure
   mode).
3. **Keyword fusion is neutral here:** after the scoping fix, hybrid ==
   semantic on these natural-language causal queries. RRF's value is exact
   identifiers (file paths, error codes); it neither helps nor hurts
   root-cause retrieval on this dataset.
4. **The pipeline's remaining gap is anchor selection,** not traversal: the
   top-1 anchor lands on a distractor in ~25% of cases, and no amount of
   walking recovers from a wrong anchor. Obvious next experiment: retrieve
   the top-3 anchor candidates and prefer ones that have causal edges —
   edge-bearing events are real decisions, distractors are leaves.

## Limitations

Synthetic, templated, and small: 4 scenario families, deterministic slot
filling, sessions of ~10–18 events, one embedding model. The numbers measure
the *mechanism* — similarity vs structure under adversarial noise — not
production performance. The honest summary is: similarity search degrades
with look-alike density exactly as the README claims, declared causality is
immune by construction, and the end-to-end pipeline inherits ~75% of that
immunity, bounded by anchor quality.
