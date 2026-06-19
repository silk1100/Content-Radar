"""SQLAlchemy ORM models — Content Radar's persistence layer.

These classes describe how data is *stored* in Postgres. They are deliberately
separate from the Pydantic models in ``models/schemas.py``:

    Pydantic (schemas.py)  ->  in-memory working objects (pipeline + API)
    SQLAlchemy (this file) ->  on-disk table definitions

Keeping them separate means each can change for its own reasons. You'll convert
between them at the boundary (e.g. ``ContentItemORM(**pydantic_item.model_dump())``)
when you write the save step — that conversion is the only place the two layers meet.
"""

import uuid
import datetime as dt
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
# We alias the *SQLAlchemy* UUID column type to ``PgUUID`` so it doesn't shadow
# Python's own ``uuid.UUID``. This collision is a classic trap: the ``Mapped[...]``
# annotation needs the *Python* type, while ``mapped_column(...)`` needs the *column* type.
from sqlalchemy.dialects.postgresql import UUID as PgUUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

# Import the same Python enums you defined for Pydantic, so the DB-level enum
# labels and the application-level values can't drift apart.
from ..models.schemas import PipelineStatus, SummaryTier

# Defined once so the two embedding columns below can never disagree. Changing
# embedding models (e.g. MiniLM -> mpnet) becomes a one-line change here plus a migration.
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 -> 384; all-mpnet-base-v2 -> 768


class Base(DeclarativeBase):
    """The declarative base every ORM model inherits from.

    SQLAlchemy collects all subclasses' table definitions onto ``Base.metadata``,
    which is what you'll later hand to ``create_all()`` to build the tables.
    """
    pass


class PipelineRun(Base):
    """One row per pipeline execution — the 'parent' in our one-to-many with content.

    Lifecycle: ``run_pipeline.py`` inserts a row with status RUNNING at the start,
    then updates the counters, cost, and status=COMPLETED (or FAILED) at the end.
    This row is what your monitoring page reads to answer "what did this run cost,
    and how many items did it process?".
    """

    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # ``server_default=func.now()`` lets Postgres stamp the time on INSERT, so the
    # value is correct even if a row is ever created outside your Python code.
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Native PG enum: ``status`` is a genuinely fixed set, so we want the database
    # itself to reject any value outside it. ``values_callable`` forces SQLAlchemy to
    # use the enum *values* ('pending', 'running', ...) as the stored labels — by
    # default it would use the *names* ('PENDING', ...) and silently mismatch Pydantic.
    status: Mapped[PipelineStatus] = mapped_column(
        Enum(PipelineStatus, name="pipeline_status",
             values_callable=lambda e: [m.value for m in e]),
        default=PipelineStatus.PENDING,
    )

    items_ingested: Mapped[int] = mapped_column(Integer, default=0)
    items_after_dedup: Mapped[int] = mapped_column(Integer, default=0)
    items_summarized: Mapped[int] = mapped_column(Integer, default=0)

    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    embedding_model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    error_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # The ORM relationship is optional for iteration 1, but it's one line and lets you
    # write ``run.items`` instead of a manual query later. ``back_populates`` keeps both
    # sides in sync in memory.
    items: Mapped[list["ContentItem"]] = relationship(back_populates="run")


class ContentItem(Base):
    """One row per piece of content — the heart of the schema.

    The pipeline fills this in stages: the adapter sets identity/content, the embed
    step sets ``embedding``, the rank step sets ``relevance_score``, and routing +
    summarization set ``summary_tier`` and ``summary``. Nullable columns are exactly
    the fields that aren't known yet at ingest time.
    """

    __tablename__ = "content_items"
    # Idempotency guard: re-running the pipeline must not insert the same post twice.
    # (source, source_id) uniquely identifies an item, so this lets you use an
    # "insert, or skip if exists" upsert later instead of accumulating duplicates.
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_source_item"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # ``source`` is Text, NOT a native enum — on purpose. Your architecture says
    # sources GROW every iteration (HN, Reddit, then ArXiv, RSS...). A native PG enum
    # would force an ``ALTER TYPE`` migration each time you add one. Text + Pydantic-side
    # validation gives you that flexibility for free. Rule of thumb: fixed set -> enum,
    # growing set -> text.
    source: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(Text)
    # FK to the run that produced this item. Nullable for now so you aren't forced to
    # create a PipelineRun before you can store anything during early experimentation.
    batch_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=True
    )

    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text, default="")  # HN stories often have none
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[dt.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # JSONB holds the per-source fields that don't fit the common shape (subreddit,
    # flair, score, descendants). ``default=dict`` gives an empty object, never NULL.
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # pgvector column. The dimension is fixed at the table level, which is why it must
    # match your embedding model — mismatched dimensions are a runtime error at insert.
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ``summary_tier`` IS a fixed set, so here a native enum is the right call.
    summary_tier: Mapped[SummaryTier] = mapped_column(
        Enum(SummaryTier, name="summary_tier",
             values_callable=lambda e: [m.value for m in e]),
        default=SummaryTier.NONE,
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped[Optional["PipelineRun"]] = relationship(back_populates="items")


class InterestProfile(Base):
    """Your interests as natural-language text plus its embedding.

    v1 is a single row: the rank step embeds ``description`` once and compares every
    ContentItem's embedding against it via cosine similarity.
    """

    __tablename__ = "interest_profile"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    description: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    # ``onupdate`` re-stamps the time whenever the row is updated, so you always know
    # when you last edited your interests.
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SourceConfig(Base):
    """Which sources are enabled and how often to fetch them.

    The pipeline reads this to decide what to pull; ``last_fetched_at`` lets you skip
    sources that were fetched recently. You won't need this until you have more than
    one source — it's defined now only because the schema is cheap to write together.
    """

    __tablename__ = "source_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_type: Mapped[str] = mapped_column(Text)  # Text for the same growth reason as ContentItem.source
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    fetch_interval: Mapped[int] = mapped_column(Integer, default=12)  # hours
    last_fetched_at: Mapped[Optional[dt.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )