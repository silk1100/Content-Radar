"""Seed (or re-seed) the single interest_profile row.

A run-once ADMIN utility, not part of the pipeline. It orchestrates three steps
— read the paragraph, embed it, upsert the row — and deliberately reuses the
capabilities it coordinates rather than reimplementing them:

  - Embedding is NOT re-done here. It calls embed_query(), so the profile vector
    and item vectors are guaranteed identical in model + normalization. If they
    diverged, every ranking would be silently meaningless.
  - Session handling is NOT re-done here. It reuses get_session(), the one place
    that owns commit / rollback / close.
  - The DB write is inline, NOT extracted into persist.py: nothing else writes
    the profile, so extracting now would be installing structure with no second
    caller to justify it. (Extract when a second caller appears, not before.)

Run from the repo root:
    python -m scripts.seed_interest_profile
    python -m scripts.seed_interest_profile --path some_other_profile.txt
"""

import argparse
import logging
from pathlib import Path

from sqlalchemy import select

from src.content_radar.db.client import get_session
from src.content_radar.db.models import InterestProfile
from src.content_radar.pipeline.embed import embed_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("interest_profile.txt")


def read_profile(path: Path) -> str:
    """Read and validate the interest paragraph. Empty file is a hard error —
    an empty profile would embed to a meaningless vector and silently break rank."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Interest profile file is empty: {path}")
    return text


def upsert_profile(session, description: str, embedding: list[float]) -> None:
    """Singleton upsert: update the one row if present, else insert it.

    interest_profile is a single-row table in v1. updated_at re-stamps itself via
    the column's onupdate, so we only touch description + embedding.
    """
    profile = session.scalars(select(InterestProfile)).first()
    if profile is None:
        session.add(InterestProfile(description=description, embedding=embedding))
        logger.info("Inserted new interest_profile row.")
    else:
        profile.description = description
        profile.embedding = embedding
        logger.info("Updated existing interest_profile row.")


def main(path: Path) -> None:
    description = read_profile(path)
    logger.info("Read profile (%d chars) from %s", len(description), path)

    embedding = embed_query(description)        # same model/normalization as items
    logger.info("Embedded profile -> %d-dim vector", len(embedding))

    with get_session() as session:              # reused commit/rollback boundary
        upsert_profile(session, description, embedding)

    logger.info("Done. interest_profile is seeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the interest_profile row.")
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_PATH,
        help="Path to the interest profile text file (default: interest_profile.txt).",
    )
    args = parser.parse_args()
    main(args.path)