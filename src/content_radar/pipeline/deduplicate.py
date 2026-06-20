"""Stage 3 — Deduplication (URL match).

The deliberately dumb V1 version: if two items share a `url`, keep the one with
the higher HN `score`, drop the rest. Items with no `url` (Ask HN / text posts)
have no shared key, so they are NEVER treated as duplicates of one another and
pass through untouched.

Fuzzy dedup — different URLs covering the same story, caught via embedding
cosine > 0.95 — is a deliberate non-goal here: items aren't embedded yet at this
stage, and URL match catches the common case while staying trivial to reason
about. That version gets extracted later, after embeddings exist.
"""

import logging
from typing import List

from ..models.schemas import ContentItem

logger = logging.getLogger(__name__)


def _engagement(item: ContentItem) -> int:
    """HN score from source_metadata, coerced to 0 when missing.

    A scoreless item is the weakest possible candidate, so 0 is the right
    default — and it keeps the comparison from blowing up on None.
    """
    score = item.source_metadata.get("score")
    return score if isinstance(score, int) else 0


def deduplicate(items: List[ContentItem]) -> List[ContentItem]:
    """Collapse same-url items to the highest-engagement one; keep url=None as-is.

    Output count is what you record as `items_after_dedup` on the run row.
    """
    best_by_url: dict[str, ContentItem] = {}
    no_url: List[ContentItem] = []

    for item in items:
        if item.url is None:
            no_url.append(item)                 # no shared key -> never a duplicate
            continue

        incumbent = best_by_url.get(item.url)
        if incumbent is None or _engagement(item) > _engagement(incumbent):
            best_by_url[item.url] = item

    deduped = list(best_by_url.values()) + no_url
    logger.info(
        "dedup: %d in -> %d out (%d url-collisions collapsed)",
        len(items), len(deduped), len(items) - len(deduped),
    )
    return deduped