"""Stage 2 — Normalization.

Converts source-specific raw dicts into the canonical Pydantic ContentItem.
After this stage, nothing downstream knows the data came from HN — that's the
entire point. Each new source gets its own normalize_* function; everything
past here (dedup, embed, rank, summarize) operates on ContentItem and never
branches on source.

Design choices (deliberate, per the V1 plan):
  - plain functions, not a class — no shared state/duplication to justify one yet
  - synchronous — pure in-memory transformation, nothing to await
  - malformed items are SKIPPED AND COUNTED, never filled with placeholder
    defaults, so `items_normalized` stays a meaningful metric
"""

import datetime as dt
import logging
from typing import List

from pydantic import ValidationError

from ..models.schemas import ContentItem, Source

logger = logging.getLogger(__name__)


def _is_valid_story(story: dict) -> bool:
    """Cheap structural gate before we attempt to build a ContentItem.

    Drops non-story items (HN `item` can be a comment/job/poll), deleted/dead
    items, and anything missing a title (a story with no title is unusable for
    embedding). The try/except in normalize_hn is the second line of defense;
    this just avoids the noise of catching obvious garbage.
    """
    return bool(
        story.get("type") == "story"
        and not story.get("deleted", False)
        and not story.get("dead", False)
        and story.get("title")
    )


def normalize_hn(story: dict) -> ContentItem | None:
    """Map one raw HN dict to a ContentItem. Returns None if malformed.

    Body uses a fallback chain: HN's own self-text (Ask HN) wins; otherwise the
    article text the adapter fetched; otherwise empty (the embedder still has
    the title). HN-specific engagement signals go into source_metadata (the
    JSONB column), NOT top-level fields — that's what keeps the shape uniform
    across sources.
    """
    try:
        return ContentItem(
            source=Source.HN,
            source_id=str(story["id"]),                 # str: pairs with source as the unique key
            title=story["title"],
            url=story.get("url"),                       # None for Ask HN / text posts
            body=story.get("text") or story.get("article_text") or "",
            author=story.get("by"),                     # Optional in schema now
            published_at=dt.datetime.fromtimestamp(     # tz-aware UTC, matches the DB column
                story["time"], tz=dt.timezone.utc
            ),
            source_metadata={
                "score": story.get("score"),
                "descendants": story.get("descendants"),
            },
        )
    except (KeyError, ValidationError) as exc:
        logger.warning("Skipping malformed HN item %s: %s", story.get("id"), exc)
        return None


def normalize_hn_batch(stories: List[dict]) -> List[ContentItem]:
    """Normalize a batch, dropping (and logging) anything malformed.

    len(output) == items_ingested - malformed_dropped, which is the number you
    record as `items_normalized` on the pipeline_runs row.
    """
    candidates = (s for s in stories if _is_valid_story(s))
    items = (normalize_hn(s) for s in candidates)
    return [item for item in items if item is not None]