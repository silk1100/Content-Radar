"""Stage 6 — Routing (tier assignment).

Pure list-slicing on the ALREADY-RANKED items: the top `deep_n` get DEEP, the
next `brief_n` get BRIEF, everything else gets NONE. This is the cost funnel —
only DEEP + BRIEF items will reach the LLM in Stage 7; the rest are stored for
browsing with zero token spend.

PRECONDITION: items must already be sorted best-first by rank_items(). Routing
by position is meaningless on an unsorted list. Routing quality is entirely
downstream of ranking quality — garbage ranking means you pay to summarize
irrelevant items, which is exactly why ranking is validated first (Checkpoint 1).
"""

import logging
from typing import List, Optional

from ..config import settings
from ..models.schemas import ContentItem, SummaryTier

logger = logging.getLogger(__name__)


def route_items(
    sorted_items: List[ContentItem],
    deep_n: Optional[int] = None,
    brief_n: Optional[int] = None,
) -> List[ContentItem]:
    """Stamp summary_tier on each item by position. Defaults to the config counts."""
    deep_n = settings.deep_summary_count if deep_n is None else deep_n
    brief_n = settings.brief_summary_count if brief_n is None else brief_n

    if sorted_items and sorted_items[0].relevance_score is None:
        logger.warning(
            "route_items got items with no relevance_score — did rank run? "
            "Tiers will be assigned by current (possibly arbitrary) order."
        )

    for i, item in enumerate(sorted_items):
        if i < deep_n:
            item.summary_tier = SummaryTier.DEEP
        elif i < deep_n + brief_n:
            item.summary_tier = SummaryTier.BRIEF
        else:
            item.summary_tier = SummaryTier.NONE

    logger.info(
        "routed %d items -> deep=%d brief=%d none=%d",
        len(sorted_items),
        min(deep_n, len(sorted_items)),
        max(0, min(brief_n, len(sorted_items) - deep_n)),
        max(0, len(sorted_items) - deep_n - brief_n),
    )
    return sorted_items