from datetime import datetime
import datetime as dt
from enum import Enum
from typing import Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---
# Enums
# ---
class Source(str, Enum):
    HN = "hn"
    REDDIT = "reddit"

class SummaryTier(str, Enum):
    DEEP = "deep"
    BRIEF = "brief"
    NONE = "none"

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class FeedbackSignal(str, Enum):
    UP = "up"
    DOWN = "down"


# ---
# Core Models
# ---

class ContentItem(BaseModel):
    # Identity
    id: UUID = Field(default_factory=uuid4)
    source: Source
    source_id: str  # Optional ID from HN or Reddit
    batch_id: Optional[UUID] = None  # Which pipeline run produced this

    # Content
    title: str
    body: str
    url: Optional[str] = None
    author: str
    published_at: datetime
    ingested_at: datetime = Field(default_factory=datetime.now(dt.timezone.utc))

    # Source-specific metadata (subredit, score, fliar, etc.)
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    # Filled by embed.py
    embedding: Optional[list[float]] = None

    # Filled by rank.py
    relevance_score: Optional[float] = None

    # Filled by route.py and summarize.py
    summary_tier: SummaryTier = SummaryTier.NONE
    summary: Optional[str] = None

class InterestProfile(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    description: str     # Natural language interest description
    embedding: Optional[list[float]]  # Embedded version of the description
    updated_at: datetime = Field(default_factory=datetime.now(dt.timezone.utc))

class PipelineRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    started_at: datetime = Field(default_factory=datetime.now(dt.timezone.utc))
    completed_at: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.PENDING

    # Counters - filled as pipeline progresses
    items_ingested: int = 0
    items_after_dedup: int = 0
    items_summarized: int = 0

    # Cost Tracking - filled after LLM calls
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0

    # Model tracking - which models were used this run
    embedding_model: Optional[str] = None
    summary_model: Optional[str] = None

    # Error tracking
    error_log: Optional[str] = None

class SourceConfig(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_type: Source
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool=True
    fetch_interval: int = 12            # Hours between fetches
    last_fetched_at: Optional[datetime] = None
    
