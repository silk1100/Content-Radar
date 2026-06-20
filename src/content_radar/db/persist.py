"""Stage 8 — Persistence.

Plain functions (no Repository class yet — nothing else writes these tables, so
there's no duplication to justify the abstraction). Responsibilities:

  1. upsert_items         — idempotent save on (source, source_id); re-runs UPDATE
                            instead of inserting duplicates.
  2. run lifecycle        — create_pipeline_run (RUNNING, first) + finalize_run
                            (COMPLETED/FAILED + metrics, last).
  3. existing_summary_keys — read companion: which items already have a summary,
                            so the summarize stage's cost guard can skip them.

All functions take an explicit `session` (the caller owns the transaction).
"""

import datetime as dt
import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..models.schemas import ContentItem as PydanticItem, PipelineStatus
from .models import ContentItem as ContentItemORM, PipelineRun

logger = logging.getLogger(__name__)


def create_pipeline_run(
    session,
    *,
    embedding_model: Optional[str] = None,
    summary_model: Optional[str] = None,
) -> PipelineRun:
    """Insert a RUNNING run row and flush so its id is available as batch_id."""
    run = PipelineRun(
        status=PipelineStatus.RUNNING,
        embedding_model=embedding_model,
        summary_model=summary_model,
    )
    session.add(run)
    session.flush()  # assigns run.id for use as batch_id below
    logger.info("pipeline_run %s started", run.id)
    return run


def upsert_items(session, items: List[PydanticItem], batch_id) -> int:
    """Idempotently save items keyed on (source, source_id). Returns rows written.

    Pre-collapses in-batch (source, source_id) collisions first: a single
    ON CONFLICT insert cannot touch the same conflict target twice (Postgres
    rejects it), and this is where 'all'-mode Ask HN duplicates would crash.

    On conflict, only the pipeline-COMPUTED fields are refreshed; identity and
    ingested_at are left untouched, so a row remembers when it was first seen.
    """
    if not items:
        return 0

    by_key: dict[tuple, PydanticItem] = {}
    for it in items:
        by_key[(it.source.value, it.source_id)] = it
    deduped = list(by_key.values())

    rows = []
    for it in deduped:
        row = it.model_dump()
        row["source"] = it.source.value   # explicit str for the Text column
        row["batch_id"] = batch_id
        rows.append(row)

    stmt = pg_insert(ContentItemORM).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_source_item",
        set_={
            "relevance_score": stmt.excluded.relevance_score,
            "summary_tier": stmt.excluded.summary_tier,
            "summary": stmt.excluded.summary,
            "embedding": stmt.excluded.embedding,
            "source_metadata": stmt.excluded.source_metadata,
            "batch_id": stmt.excluded.batch_id,
        },
    )
    session.execute(stmt)
    logger.info("upserted %d items (batch %s)", len(rows), batch_id)
    return len(rows)


def existing_summary_keys(session) -> set[tuple[str, str]]:
    """Return {(source, source_id), ...} for items that ALREADY have a stored
    summary. The summarize stage skips these so a re-run never re-bills work
    that's already done — the single most expensive bug this pipeline could ship.
    """
    rows = session.execute(
        select(ContentItemORM.source, ContentItemORM.source_id)
        .where(ContentItemORM.summary.isnot(None))
    ).all()
    return {(src, sid) for src, sid in rows}


def finalize_run(
    session,
    run: PipelineRun,
    *,
    status: PipelineStatus,
    items_ingested: int = 0,
    items_after_dedup: int = 0,
    items_summarized: int = 0,
    total_tokens_used: int = 0,
    total_cost_usd: float = 0.0,
    error_log: Optional[str] = None,
) -> None:
    """Stamp the run's terminal state + metrics. `run` must be attached to this
    session (it is, when created via create_pipeline_run in the same session)."""
    run.status = status
    run.completed_at = dt.datetime.now(dt.timezone.utc)
    run.items_ingested = items_ingested
    run.items_after_dedup = items_after_dedup
    run.items_summarized = items_summarized
    run.total_tokens_used = total_tokens_used
    run.total_cost_usd = total_cost_usd
    run.error_log = error_log
    logger.info("pipeline_run %s -> %s", run.id, status.value)