# Similarity Search Can't Debug Your Agent

*A draft write-up of the Hippocampus project: the thesis, the system, the
eval, and what the eval broke. Edit freely — the numbers are reproducible
from the repo (`python -m evals.run --provider fastembed --sessions 40
--seed 42`); everything else is prose.*

---

## The problem

When an AI agent fails, the question you actually ask is *"what led to
this?"* — and the tool most systems reach for answers a different question:
*"what looks like this?"*

Standard RAG retrieves by embedding similarity. That's the right tool for
"find me things about X." It's the wrong tool for debugging, because the
events most similar to a failure are other failure-shaped things: the stale
ticket about the same exception, the troubleshooting doc you wrote about it,
last month's similar crash in a different service. The actual root cause — a
null check removed as an "optimization" three days earlier — shares almost no
vocabulary with the crash it produced. Similarity search structurally cannot
rank it first, and it structurally must rank the look-alikes above it.

Hippocampus is a small exploration of the alternative: treat the agent's
decision log as an event-sourced record, let producers declare causality, and
retrieve by *walking backward from the failure* instead of searching for its
twins.

## The system, in one diagram

```
 Agent publishes decisions              Postgres (pg_kafka + pgvector)
 (fire-and-forget, Kafka protocol)     ┌────────────────────────────────┐
                                       │ kafka.messages   (event log,   │
 {"id": "d2",                          │   total order via global_offset)│
  "parent_id": "d1",     ──produce──▶  │        │                        │
  "context": "...",                    │        ▼ consumer               │
  "reasoning": "...",                  │ hippocampus.embeddings          │
  "action": "..."}                     │   (vectors + text + anchors)    │
                                       │ hippocampus.causal_edges        │
        ▲                              │   (parent_id resolved to        │
        │ MCP tools (find_similar,     │    offset → offset edges)       │
 Agent reads via ─────────────────────▶│                                 │
 replay_causal_chain, ...)             └────────────────────────────────┘
```

No Kafka cluster, no vector database — one Postgres with two extensions
([pg_kafka](https://github.com/RTrentJones/pg_kafka), pgvector). Producers
can't know broker offsets at publish time, so causality is declared with
producer-assigned IDs (`parent_id`); the embedding consumer resolves them to
offsets and stores explicit edges. `replay_causal_chain` walks those edges
with a recursive CTE, falling back to a plain temporal window when no edges
were declared — and tagging every result with which guarantee it carried.

## The claim is measurable, so measure it

The claim — *walking back beats searching for look-alikes* — is a testable
statement about retrieval, so the repo ships an eval instead of an argument.

### The dataset is a trap, on purpose

Each synthetic session plants a causal chain and buries it under distractors
engineered to share the **failure's** vocabulary but not the **cause's**:

```
 one session, in log order                              lexical overlap
 ────────────────────────────────────────────────────── with the query
  ROOT CAUSE  "remove null check as an optimization"        LOW   ◁─┐
  noise       "publish on-call schedule"                    none    │
  CHAIN       "deploy release with the change"              LOW   ◁─┤ parent_id
  DISTRACTOR  "close stale NullPointerException ticket"     HIGH     ✗ no edge
  CHAIN       "error spike for expired sessions"            MED   ◁─┤
  DISTRACTOR  "update guide for token validation crashes"   HIGH     ✗ no edge
  FAILURE     "crash with NullPointerException"             HIGH  ◁─┘

  query:        "Why did the server crash with a
                 NullPointerException in token validation?"
  ground truth: the ROOT CAUSE event
```

The query describes the failure. The distractors match the query. The root
cause doesn't. If similarity search can win here, the thesis is wrong — and
the eval README says exactly that: a negative result goes in the write-up too.

### Five ways to answer the same question

| arm | mechanism | given |
|---|---|---|
| `semantic` | top-k by embedding cosine | query only |
| `hybrid` | vector + keyword RRF | query only |
| `temporal` | the k events before the failure | query + true anchor (oracle) |
| `causal` | walk `parent_id` edges back from the failure | query + true anchor (oracle) |
| `pipeline` | search finds the anchor, then walk | query only — the real system |

The oracle arms isolate the mechanism; the pipeline arm is the honest
end-to-end test, where the system must find its own starting point.

## Results

40 sessions, `all-MiniLM-L6-v2` embeddings, retrieval scoped to the
incident's session, seed 42. Root-cause recall@5 as look-alike distractors
per session grow:

```
             d=2                  d=6                  d=12
  causal      ████████████ 1.00    ████████████ 1.00    ████████████ 1.00
  pipeline    ████████████ 1.00    █████████    0.75    █████████    0.75
  semantic    ████████████ 1.00    █████        0.43    █            0.10
  temporal    ██████████   0.85    ████▌        0.38    █            0.10
```

Three readings:

1. **With little noise, everything works** (d=2 column). Similarity search is
   fine when there's nothing that looks like the failure except the failure's
   own history. Distractors are the variable that breaks it — which is the
   thesis stated as an ablation.
2. **Similarity and temporal retrieval decay with noise; the causal walk
   doesn't.** Edges don't get noisier when the log does. Distractors have no
   edges, so they are structurally invisible to the traversal — 1.000 recall
   at every density, and the best MRR (0.31) because the cause *ranks first*
   instead of merely appearing.
3. **The full pipeline inherits most of that immunity** — flat at 0.750 from
   d=6 onward. Its only failure mode is the anchor step: when the initial
   search lands on a distractor (an event with no edges), no amount of
   walking recovers. The obvious next experiment is edge-aware anchor
   re-ranking: real decisions have edges, distractors are leaves.

## The part I didn't expect: the eval found bugs

Building a measurement instrument for your own system is uncomfortable in
the best way. Two defects survived a 100+-test suite and fell out of the
eval within hours:

**1. Scoped hybrid search was returning near-arbitrary rows.** The RRF
implementation selected its vector and keyword candidate pools globally,
*then* applied the topic filter — so a search scoped to one session drew its
candidates from every session and kept whatever leftovers matched the scope.
Symptom in the data: hybrid's top-1 anchor landed on irrelevant noise 57% of
the time while plain vector search (which filters before ranking) hit the
failure event 75% of the time. One diagnostic script, one SQL fix — filter
inside the candidate CTEs — and the pipeline's end-to-end recall went 0.275
→ 0.750. The unit tests never caught it because they never combined scoping
with competing candidates.

**2. A vector index that loses data.** The original migration created an
ivfflat index on the *empty* embeddings table. ivfflat trains its centroids
at `CREATE INDEX` time; trained on nothing, they're random, and a query may
probe a list that simply doesn't contain the row you just inserted. First
symptom: recall@10 measured *below* recall@5 — impossible for a fixed
ranking. Reduced to its minimal case: a one-row table where `ORDER BY
embedding <=> $1 LIMIT 1` returned nothing. The fix was to not build an ANN
index on an empty table at all — exact scans are correct at this scale, and
HNSW (which builds incrementally, no training step) is the graduation path.

Neither bug is exotic; both are the kind of thing that silently degrades a
production RAG system while every test stays green. The retrieval *quality*
eval is what made them visible, because it measures outcomes rather than
plumbing.

## Honest limitations

The dataset is synthetic, templated, and small: four scenario families,
deterministic slot-filling, sessions of 10–18 events, one embedding model.
These numbers measure a *mechanism* — similarity vs. structure under
adversarial noise — not production performance. Causal retrieval's perfect
score is also partly by construction: it presumes producers declared
causality, which is a real integration cost the similarity baseline doesn't
pay. The interesting production questions — do agents reliably emit
`parent_id`? what fraction of real failure chains are connected? — need real
traces, which is the next phase: a Claude Code hook publishing decisions,
and the agent debugging its own prior sessions through the MCP tools.

## Takeaways

- For agent debugging, **retrieval should follow structure where structure
  exists** and fall back to similarity only for finding the entry point.
- **Eval your retrieval, not just your code.** Correctness tests verify that
  queries run; only a quality eval verifies they return the *right* things —
  and ours paid for itself in found bugs before producing a single
  publishable number.
- **Minimal infrastructure kept the experiment cheap.** The whole system —
  broker, vectors, graph, eval — is one Postgres. Everything here graduates
  (real Kafka, HNSW, a graph store) without changing the MCP interface
  agents talk to.

*Code, eval harness, and per-run JSON artifacts:
[github.com/RTrentJones/Hippocampus](https://github.com/RTrentJones/Hippocampus).*
