# Content Radar — System Design & Implementation Plan

## Project Overview

Content Radar is a personal AI-powered content curation system. It automatically pulls content from sources you care about (Hacker News, Reddit, and eventually ArXiv, RSS feeds, YouTube, podcasts), ranks everything by relevance to your interests, generates summaries at varying depth, and delivers a personalized daily digest.

The project serves as a comprehensive AI engineering curriculum, covering: embeddings & retrieval, prompt engineering, model routing, evaluation, cost optimization, monitoring, guardrails, and personalization.

---

## AI Engineering Concepts by Pipeline Stage

| Pipeline Stage | AI Engineering Concepts |
|---|---|
| Embed & Rank | Embeddings, vector similarity, retrieval, reranking |
| Filter & Route | Model routing, cost funnels, tiered processing |
| Summarize | Prompt engineering, structured outputs, prompt templates per content type |
| Feedback Loop | Personalization, learned interest profiles, preference learning |
| Deduplication | Embedding similarity for fuzzy matching |
| Monitoring | Observability, quality tracking, cost dashboards, drift detection |
| Evaluation | LLM-as-judge, precision@k, retrieval metrics, automated evals |
| Cluster Boost | Trending topic detection via embedding clustering |

---

## Architecture

### Components

**Pipeline (Python)** — A batch job triggered by a scheduler (cron). Runs every 6-12 hours. Pulls content, processes it through all stages, writes results to the database, and exits. This is a standalone process, not a web server.

**API (FastAPI)** — A thin layer between the database and the frontend. Serves the digest, accepts feedback, exposes monitoring data, and optionally triggers pipeline runs. Contains no business logic beyond CRUD, filtering, and pagination.

**Frontend (Next.js)** — The user-facing interface. Digest reading, feedback submission, source configuration, and monitoring dashboards. Talks only to the API.

**Key Principle:** The pipeline and the API are independent processes. They communicate entirely through the database. The pipeline is a producer, the API+frontend is a consumer, the database is the shared state.

### Data Flow

```
[Cron Scheduler]
      │
      ▼
[Pipeline (Python)]
      │
      ├── pulls from ──→ [HN API]
      ├── pulls from ──→ [Reddit API]
      ├── embeds via ──→ [HuggingFace / Embedding Model]
      ├── summarizes via → [LLM API (Claude/GPT)]
      │
      ▼
[Supabase (Postgres + pgvector)]
      ▲
      │
[FastAPI Server] ◄──── HTTPS ────► [Next.js Frontend (Vercel)]
```

---

## Pipeline Stages (Detail)

### Stage 1 — Ingest

Pull raw content from sources on a schedule. Each source has its own adapter implementing a common interface (`fetch_new_items()`). The adapter pattern allows adding new sources without modifying any downstream code.

**HN API:**
- `https://hacker-news.firebaseio.com/v0/topstories.json` — up to 500 item IDs
- `https://hacker-news.firebaseio.com/v0/item/{id}.json` — individual item details
- No authentication required
- v1 approach: rank based on title + comments, not the linked article

**Reddit API:**
- REST API with OAuth2 authentication (register an app at reddit.com/prefs/apps)
- Subreddits for v1: r/MachineLearning, r/LocalLLaMA, r/Python, r/datascience (configurable)
- Returns structured JSON with self-text, scores, metadata

**Expected volume:** ~300-500 items per batch across both sources.

### Stage 2 — Normalize

Convert source-specific data into a common `ContentItem` format. After this stage, the rest of the pipeline is source-agnostic. See the database schema below for the canonical shape.

### Stage 3 — Deduplicate

Prevent the same content from appearing multiple times in the digest.

**v1 approach:** URL matching — if two items link to the same URL, keep the one with higher engagement (score/comments).

**Future approach:** Embedding similarity — if two items have cosine similarity > 0.95, treat them as duplicates regardless of URL. This catches cases where different sources link to different coverage of the same story.

### Stage 4 — Embed & Rank

The core AI stage. Convert each content item and the user's interest profile into vectors, compute similarity scores.

**Interest profile (v1):** A natural language paragraph describing your interests, embedded as a single vector. Example: "I'm a data scientist interested in deep learning, PyTorch, FastAPI, building SaaS products, MLOps, LLM applications, AI engineering, embeddings, and retrieval-augmented generation. I want practical techniques, not theoretical papers."

**Embedding models to experiment with:**
- HuggingFace sentence-transformers (free, local): `all-MiniLM-L6-v2` (fast), `all-mpnet-base-v2` (better quality)
- OpenAI Embeddings API (cheap, high quality, API dependency)
- Cohere Embed API (similar tradeoff to OpenAI)

**Ranking:** Cosine similarity between the interest profile embedding and each item's embedding. Higher similarity = higher relevance score.

**Cluster boost (future):** After initial ranking, detect clusters of similar articles using DBSCAN or k-means over embeddings. If multiple moderately-relevant articles cluster together, boost their scores — this signals a trending topic.

**Storage:** Embeddings stored in pgvector columns in Supabase. Similarity search via SQL: `SELECT * FROM content_items ORDER BY embedding <-> target_embedding LIMIT k;`

### Stage 5 — Filter & Route

Cost funnel — not every item deserves an expensive LLM call.

| Tier | Criteria | Processing | Model |
|---|---|---|---|
| Deep | Top 10 by relevance | Full summary with personal relevance explanation | Strong model (Claude/GPT-4) |
| Brief | Items 11-30 | One-sentence summary | Cheap/fast model (Haiku/GPT-4o-mini) |
| None | Items 31+ | Title and score only, stored for browsing | No LLM call |

Thresholds and tier sizes are configurable. Routing quality depends on ranking quality — if ranking is poor, expensive summarization is wasted on irrelevant items.

### Stage 6 — Summarize

Generate summaries using tier-appropriate prompts.

**Deep summary prompt requirements:** explain why this is relevant to the user specifically, extract key takeaways, note anything actionable, focus on practical implications.

**Brief summary prompt requirements:** capture the gist in one sentence, include the core topic and why it might matter.

Prompt design is iterative — expect to refine prompts over days/weeks based on output quality.

### Stage 7 — Deliver

Produce the digest and make it available via the API. The frontend renders it.

### Stage 8 — Feedback

User reactions (thumbs up/down) stored in the database. Feedback is used to:
- Evaluate ranking quality (are highly-ranked items getting thumbs up?)
- Adjust the interest profile over time (v2+)
- Track preference drift

### Stage 9 — Monitor

Track per-run metrics: items ingested, items after dedup, items summarized, tokens used, cost in USD, latency, error rates. Power the monitoring dashboard and detect quality degradation.

---

## Database Schema (Supabase / PostgreSQL + pgvector)

### `content_items`

| Column | Type | Notes |
|---|---|---|
| id | uuid, PK | Generated |
| source | enum ('hn', 'reddit') | Extensible as sources are added |
| source_id | string | Original post ID from the source |
| url | string, nullable | External link if any |
| title | string | |
| body | text | Self-text, comment thread, etc. |
| author | string | |
| source_metadata | jsonb | Source-specific fields (subreddit, score, flair for Reddit; score, descendants for HN). JSONB because each source has a different shape |
| published_at | timestamp | When posted on the source |
| ingested_at | timestamp | When our pipeline pulled it |
| embedding | vector (pgvector) | Filled during ranking stage |
| relevance_score | float, nullable | Cosine similarity to interest profile |
| summary_tier | enum ('deep', 'brief', 'none') | Determined by routing stage |
| summary | text, nullable | Generated summary |
| batch_id | uuid, FK → pipeline_runs | Which pipeline run produced this |

### `interest_profile`

| Column | Type | Notes |
|---|---|---|
| id | uuid, PK | |
| description | text | Natural language interest description |
| embedding | vector | Embedded version of the description |
| updated_at | timestamp | |

v1: single row. Evolves into learned profiles with feedback.

### `feedback`

| Column | Type | Notes |
|---|---|---|
| id | uuid, PK | |
| content_item_id | uuid, FK → content_items | |
| signal | enum ('up', 'down') | Extensible to 'more_like_this', 'hide_topic', etc. |
| created_at | timestamp | |

### `pipeline_runs`

| Column | Type | Notes |
|---|---|---|
| id | uuid, PK | |
| started_at | timestamp | |
| completed_at | timestamp | |
| status | enum ('running', 'completed', 'failed') | |
| items_ingested | int | |
| items_after_dedup | int | |
| items_summarized | int | |
| total_tokens_used | int | |
| total_cost_usd | float | |
| embedding_model | string | Which model was used this run |
| summary_model | string | |
| error_log | text, nullable | |

### `source_configs`

| Column | Type | Notes |
|---|---|---|
| id | uuid, PK | |
| source_type | enum ('hn', 'reddit') | |
| config | jsonb | e.g. `{"subreddit": "MachineLearning"}` or `{"endpoint": "topstories"}` |
| enabled | boolean | |
| fetch_interval | int | Hours between fetches |
| last_fetched_at | timestamp | |

### Why PostgreSQL (Not NoSQL or Graph DB)

**Not NoSQL:** 90% of the data is uniform and relational (title, body, embedding, scores, timestamps). Only `source_metadata` varies by source, and JSONB handles that cleanly. Choosing an entire database paradigm for one flexible field isn't justified. Additionally, pgvector lives in PostgreSQL — using MongoDB would require a separate vector database alongside it.

**Not Graph DB:** Relationships between articles (similarity, same-topic clusters) are implicit and continuous, encoded directly in embedding space. Cosine similarity gives you "articles related to this article" without explicit edges. Graph databases (Neo4j) shine when you have explicit, typed, multi-hop relationships (citations, authorship networks, funding chains). This project has similarity relationships, which vectors handle natively.

---

## API Endpoints

```
GET  /digest              — Today's ranked, summarized content
GET  /digest/history      — Past digests by date
POST /feedback            — Submit thumbs up/down for an item
GET  /stats               — Pipeline run history, costs, metrics
GET  /sources             — List configured sources
POST /sources             — Add/modify a source
POST /pipeline/trigger    — Manually kick off a pipeline run
```

---

## Frontend Pages

**Digest Page (main screen):** Today's top content ranked by relevance. Deep summaries expanded, brief summaries collapsed with click-to-expand. Each item has thumbs up/down buttons, link to original, and metadata (source, score, time).

**Settings Page:** Edit interest profile text, manage sources (enable/disable subreddits, adjust fetch frequency), set preferences (digest size, summary depth).

**Monitoring Page:** Charts showing cost per run over time, items processed per run, feedback patterns (thumbs-down rate as a proxy for ranking quality), cost breakdown by model.

---

## Deployment

| Component | Platform | Notes |
|---|---|---|
| Pipeline | Heroku Scheduler or Linode cron | Runs every 6-12 hours, executes and exits |
| API | Heroku or Linode | Continuous FastAPI process |
| Frontend | Vercel | Free tier, natural home for Next.js |
| Database | Supabase | Free tier handles this scale |

---

## Project Structure (Python)

```
content-radar/
    adapters/
        base.py          # Abstract base adapter interface
        hn.py            # Hacker News adapter
        reddit.py        # Reddit adapter
    pipeline/
        normalize.py     # Convert raw data to ContentItem
        deduplicate.py   # Remove duplicate content
        embed.py         # Generate embeddings
        rank.py          # Compute relevance scores
        route.py         # Assign summary tiers
        summarize.py     # Generate summaries via LLM
    models/
        schemas.py       # Pydantic models (ContentItem, Feedback, etc.)
    monitoring/
        tracker.py       # Log pipeline run metrics
    config.py            # Settings, API keys, model choices
    run_pipeline.py      # Entry point — orchestrates all stages
```

---

## Build Phases

### Phase 1 — Core Pipeline (Stages 1-4)
Ingest → Normalize → Deduplicate → Rank. No API, no frontend. Run manually, verify results by querying Supabase directly.

### Phase 2 — Summarization (Stages 5-6)
Add routing tiers and LLM summarization. Still no frontend. Read digests from the database.

### Phase 3 — API + Basic Frontend
FastAPI endpoints + Next.js digest page. First time seeing results in a UI.

### Phase 4 — Feedback Loop
Thumbs up/down in the frontend, wired to the API. Begin using feedback to evaluate and adjust rankings.

### Phase 5 — Monitoring & Settings
Dashboard for costs, quality metrics, pipeline history. Settings page for sources and interest profile.

### Phase 6 — Advanced Improvements
- Experiment with different embedding models and compare retrieval quality
- Learned interest profiles (update embeddings based on feedback patterns)
- Cluster-based trending topic detection
- Additional sources (ArXiv, RSS, YouTube transcripts, podcasts)
- Sentiment/trend analysis across topics over time
- Reranking with cross-encoders or LLM-based rerankers

---

## Future Extensions

**Summarizer depth:** Add a "deep dive" mode that fetches the linked article (for HN) or full paper (for ArXiv) and produces a comprehensive analysis.

**Trend analyst:** Track how public opinion shifts on topics over weeks by comparing embedding distributions and sentiment across time windows.

**Anti-doomscrolling:** The system replaces social media browsing with a curated, finite, high-signal digest. Feedback refines it over time so it gets better the more you use it.