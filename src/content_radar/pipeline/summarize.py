"""Stage 7 — Summarization.

Provider-agnostic via the OpenAI-compatible SDK: the same client + code talks to
Ollama (local, free) or DeepSeek (hosted) — only config changes. Only DEEP and
BRIEF items are summarized; NONE items never reach the LLM (the cost funnel).

Key safeguards:
  - cost guard (caller-supplied skip_keys): items already summarized in the DB
    are skipped, so re-runs never re-bill work that's done.
  - failure isolation: one failed call leaves summary=None, ticks a counter, and
    the loop continues — a crashed mid-run is far worse than a missing summary.

Prompts live here as constants for V1; extract to a prompts/ package when you
start iterating on wording heavily.
"""

import logging
from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

from openai import OpenAI

from ..config import settings
from ..models.schemas import ContentItem, SummaryTier

logger = logging.getLogger(__name__)

# OpenAI-compatible client — same object for Ollama or DeepSeek, set by config.
_client = OpenAI(base_url=settings.llm_url, api_key=settings.llm_key)

# Cap the article body fed into the prompt: LLMs have big context windows, but a
# 50k-char article is mostly tokens you pay for and don't need.
MAX_BODY_CHARS = 4000

# USD per 1,000,000 tokens, (input, output). Local models cost nothing. Verify
# hosted rates before relying on the dollar figure — providers change pricing.
PRICING = {
    "granite4.1:8b": (0.0, 0.0),
    "deepseek-v4-flash": (0.14, 0.28),
}

_PERSONA = (
    "a data scientist and ML engineer interested in applied machine learning, "
    "LLMs, RAG, embeddings, MLOps, and the engineering of shipping ML systems"
)

DEEP_PROMPT = """You are curating a personalized tech digest. The reader is {persona}.

Summarize the item below for them in about 120 words: what it is, why it might
matter to someone with those interests, the key takeaways, and anything
actionable. Be concrete; skip filler and hype.

Title: {title}
Link: {url}
Content: {body}
"""

BRIEF_PROMPT = """You are curating a tech digest for {persona}.

In ONE sentence, capture the gist of the item below and why it might matter.

Title: {title}
Content: {body}
"""


@dataclass
class SummarizeMetrics:
    items_summarized: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    failures: int = 0
    skipped: int = 0           # already-summarized (cost guard) + nothing to do


def _summarize_one(item: ContentItem) -> Tuple[str, int, int]:
    """One LLM call. Returns (summary, prompt_tokens, completion_tokens)."""
    template = DEEP_PROMPT if item.summary_tier == SummaryTier.DEEP else BRIEF_PROMPT
    prompt = template.format(
        persona=_PERSONA,
        title=item.title,
        url=item.url or "—",
        body=(item.body or "")[:MAX_BODY_CHARS],
    )
    resp = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    summary = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    prompt_tok = usage.prompt_tokens if usage else 0
    completion_tok = usage.completion_tokens if usage else 0
    return summary, prompt_tok, completion_tok


def summarize(
    items: List[ContentItem],
    skip_keys: FrozenSet[Tuple[str, str]] = frozenset(),
) -> Tuple[List[ContentItem], SummarizeMetrics]:
    """Summarize DEEP/BRIEF items not already done. Mutates summaries in place.

    skip_keys is the set of (source, source_id) that already have a stored summary
    (from the DB). This is the COST GUARD: those items are not re-summarized, so a
    re-run never pays twice for the same content.
    """
    metrics = SummarizeMetrics()

    candidates = [it for it in items if it.summary_tier != SummaryTier.NONE]
    targets = [it for it in candidates if (it.source.value, it.source_id) not in skip_keys]
    metrics.skipped = len(candidates) - len(targets)

    in_rate, out_rate = PRICING.get(settings.llm_model, (0.0, 0.0))

    for item in targets:
        try:
            summary, prompt_tok, completion_tok = _summarize_one(item)
            item.summary = summary
            metrics.items_summarized += 1
            metrics.total_tokens += prompt_tok + completion_tok
            metrics.total_cost_usd += (prompt_tok / 1e6) * in_rate + (completion_tok / 1e6) * out_rate
        except Exception as exc:  # failure isolation — one bad call mustn't abort the run
            logger.warning("summarize failed for %s (%s): %s", item.source_id, item.title[:40], exc)
            metrics.failures += 1

    logger.info(
        "summarized=%d skipped=%d failed=%d tokens=%d cost=$%.4f",
        metrics.items_summarized, metrics.skipped, metrics.failures,
        metrics.total_tokens, metrics.total_cost_usd,
    )
    return items, metrics