"""Stage 5 — Ranking.

Score every item by cosine similarity to the interest profile, sort highest
first. Because all vectors were L2-normalized at embed time (||v|| = 1), cosine
similarity reduces to a plain dot product — so ranking the whole batch is one
matrix multiply: (N, 384) @ (384,) -> (N,).

Why NumPy-in-Python and not a pgvector SQL query for V1:
  - at a few hundred items it's instant;
  - you can see and debug the actual math;
  - items aren't saved yet at rank time — they're in-memory ContentItems, so
    ranking here keeps the pipeline a straight line of object transformations.
You move to `ORDER BY embedding <-> profile` in SQL when the dataset is large.

This stage is pure: it loads nothing and writes nothing on its own. The caller
hands it the items and the already-seeded profile vector.
"""

import logging
from typing import List

import numpy as np
from sqlalchemy import select

from ..db.client import get_session
from ..db.models import InterestProfile
from ..models.schemas import ContentItem

logger = logging.getLogger(__name__)


def load_interest_profile_embedding() -> List[float]:
    """Read the seeded profile vector from the DB. Does NOT re-embed — the seed
    script already did that with the same model, which is the whole point."""
    with get_session() as session:
        profile = session.scalars(select(InterestProfile)).first()

    if profile is None or profile.embedding is None:
        raise RuntimeError(
            "No seeded interest_profile found. Run: "
            "python -m scripts.seed_interest_profile"
        )
    return list(profile.embedding)


def rank_items(
    items: List[ContentItem], profile_embedding: List[float]
) -> List[ContentItem]:
    """Set each item's relevance_score and return the list sorted highest-first.

    With unit-length vectors this is a single dot product per item — computed
    for the whole batch as one matrix multiply.
    """
    if not items:
        return items

    profile = np.asarray(profile_embedding, dtype=np.float32)          # (384,)
    matrix = np.asarray([it.embedding for it in items], dtype=np.float32)  # (N, 384)

    scores = matrix @ profile          # (N,) cosine scores, since all are unit-length

    for it, score in zip(items, scores):
        it.relevance_score = float(score)

    items.sort(key=lambda it: it.relevance_score, reverse=True)        # highest first

    # Sanity signal: if the top scores aren't meaningfully above the tail, the
    # profile is too vague or something upstream is off — investigate before
    # trusting the routing that depends on this ordering.
    s = scores
    logger.info(
        "ranked %d items | score max=%.3f median=%.3f min=%.3f",
        len(items), float(s.max()), float(np.median(s)), float(s.min()),
    )
    return items