"""LLM-as-judge: can a model identify the root cause from retrieved context?

The retrieval metrics measure whether the true cause was *surfaced*; this
measures whether the retrieved context is *sufficient* for a model to name it.
Requires the `anthropic` package and an ANTHROPIC_API_KEY (install with
`pip install -e ".[evals]"`); the eval harness runs without it.
"""

import logging

from pydantic import BaseModel

from .dataset import Decision, EvalCase

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

_PROMPT = """You are debugging an AI agent. The agent hit this failure:

<failure_query>
{query}
</failure_query>

A retrieval system returned these prior agent decisions as context (most
relevant first). Each has an id:

<retrieved_context>
{context}
</retrieved_context>

Identify the single decision that is the ROOT CAUSE of the failure — the
earliest event that set the failure in motion, not merely a symptom or a
similar-sounding but unrelated event. If none of the retrieved decisions is
the root cause, say so."""


class Verdict(BaseModel):
    root_cause_id: str | None
    reasoning: str


def _render_context(decisions: list[Decision]) -> str:
    blocks = []
    for decision in decisions:
        blocks.append(
            f"[id: {decision.id}]\n"
            f"context: {decision.context}\n"
            f"reasoning: {decision.reasoning}\n"
            f"action: {decision.action}"
        )
    return "\n\n".join(blocks)


def judge_case(
    client,
    case: EvalCase,
    retrieved: list[Decision],
    model: str = DEFAULT_JUDGE_MODEL,
) -> bool:
    """Ask the judge to name the root cause; return whether it got it right."""
    if not retrieved:
        return False

    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _PROMPT.format(query=case.query, context=_render_context(retrieved)),
            }
        ],
        output_format=Verdict,
    )
    verdict = response.parsed_output
    correct = verdict.root_cause_id == case.root_cause_id
    if not correct:
        logger.debug(
            f"Judge picked {verdict.root_cause_id!r}, truth {case.root_cause_id!r}: "
            f"{verdict.reasoning[:120]}"
        )
    return correct


def make_client():
    """Create an Anthropic client, or explain what's missing."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "LLM judge requires the anthropic package: pip install -e '.[evals]'"
        ) from e
    return anthropic.Anthropic()
