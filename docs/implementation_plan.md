# Content Radar — V1 Implementation Plan (Pipeline Core)

This plan covers **only** the offline pipeline: ingest content from Hacker News,
embed it, rank it against your interests, route it into summary tiers, summarize
the top items with an LLM, and persist everything to Supabase — orchestrated by a
single linear `run_pipeline.py`. FastAPI and the frontend are deliberately a
separate plan, to be built later.

The sequencing principle for V1: **ingest → normalize → dedup → embed → rank → save
is a verifiable checkpoint before any LLM call happens.** You run the pipeline once
with summarization disabled, open Supabase, and confirm the rankings match your gut.
Only then do you turn on summarization and start spending tokens. This keeps the
"thin slice first, widen later" discipline even though V1 is a fuller slice than a
true MVP.

---

## 1. Scope & Non-Goals

**In scope (V1):**

- Ingestion from Hacker News only (no auth required).
- Normalization of raw HN data into the `ContentItem` shape.
- URL-based deduplication.
- Embedding via a local sentence-transformer.
- Ranking by cosine similarity against a seeded interest profile.
- Routing into `deep` / `brief` / `none` tiers by rank position.
- Summarization of `deep` and `brief` items via an OpenAI-compatible LLM
  (Ollama **or** DeepSeek — selected by config, no code difference).
- Idempotent persistence to Supabase, plus a `pipeline_runs` metrics row.
- A linear `run_pipeline.py` that calls each stage in order.

**Explicitly deferred (these are decisions, not oversights):**

- FastAPI and the Next.js frontend — separate plan.
- Reddit and all other sources — the next increment after V1 works end to end.
- Alembic migrations — `create_all()` is correct while the schema is still molten.
- A pgvector index (ivfflat/hnsw) — a sequential scan is *faster* at a few hundred
  rows; the index earns its place in the tens of thousands.
- Embedding-based (fuzzy) dedup, clustering, learned interest profiles, reranking.
- The Adapter / Strategy / Repository / Orchestrator abstractions. V1 stays a flat
  script of plain functions. Each pattern gets **extracted later**, when the
  duplication or rigidity it solves actually shows up.

---

## 2. Prerequisites & Environment

### 2.1 Config additions (`config.py`)

You already have `supabase_url`, `supabase_key`, `embedding_model`, the Reddit
fields, and the pipeline counts. For V1 you need these, with two corrections from
earlier:

```python
    # --- Database (raw Postgres for SQLAlchemy; NOT the same as supabase_url/key) ---
    database_url: str

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"   # 384-dim; must match EMBEDDING_DIM

    # --- LLM (OpenAI-compatible: works for both Ollama and DeepSeek) ---
    llm_base_url: str = "http://localhost:11434/v1"   # Ollama default
    llm_api_key: str = "ollama"                       # any string for Ollama; real key for DeepSeek
    llm_model: str = "llama3.2"                        # or "deepseek-v4-flash"
```

> **Correction carried over from the schema step:** rename the old `model_key` to
> `llm_api_key`. Anything starting with `model_` collides with Pydantic v2's
> protected namespace and will warn/misbehave.

To switch providers you change three values and nothing else:

| Setting        | Ollama (local, free)            | DeepSeek (hosted)            |
|----------------|---------------------------------|------------------------------|
| `llm_base_url` | `http://localhost:11434/v1`     | `https://api.deepseek.com`   |
| `llm_api_key`  | `"ollama"` (ignored)            | your real DeepSeek key       |
| `llm_model`    | e.g. `llama3.2`, `qwen3:8b`     | `deepseek-v4-flash`          |

### 2.2 `.env`

```
DATABASE_URL=postgresql+psycopg2://postgres.[PROJECT_REF]:[PASSWORD]@aws-[REGION].pooler.supabase.com:5432/postgres?sslmode=require
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
```

### 2.3 Dependencies

```
sqlalchemy
psycopg2-binary
pgvector
pydantic
pydantic-settings
httpx                 # async HTTP for the HN fan-out
sentence-transformers # local embeddings
numpy                 # cosine similarity
openai                # OpenAI-compatible client for Ollama/DeepSeek
```

### 2.4 Database prerequisites (already done in the previous step)

- `create extension if not exists vector;` enabled in Supabase.
- Tables created via `python -m content_radar.db.init_db`.

### 2.5 The dimension lock — read this before choosing a model

Your ORM has `EMBEDDING_DIM = 384`, which is hard-wired into the `Vector(384)`
columns. `all-MiniLM-L6-v2` outputs 384 dimensions, so it matches. If you ever
switch to `all-mpnet-base-v2` (768 dims), you must change `EMBEDDING_DIM` **and**
recreate the table — pgvector rejects a vector whose length doesn't match the
column. This is exactly why the dimension is a single named constant: the mismatch
is a runtime insert error, not a quiet bug. For V1, stay on MiniLM/384.

---

## 3. The Two Model Layers: Pydantic ↔ ORM *(confusing step — expanded)*

You now have **two** `ContentItem`-shaped classes and this is correct, not
redundant. They answer different questions:

- `models/schemas.py` → **Pydantic `ContentItem`**: the in-memory working object.
  It flows through every pipeline stage. Each stage fills in more of it
  (`embedding`, then `relevance_score`, then `summary_tier`/`summary`).
- `db/models.py` → **SQLAlchemy `ContentItem`**: a table row. It describes how the
  object is stored on disk.

**They meet exactly once: at save time.** Everything upstream (ingest → summarize)
works with the Pydantic object. The persistence stage is the *only* place that
converts Pydantic → ORM:

```python
# At the save boundary only:
orm_item = ContentItemORM(**pydantic_item.model_dump())
```

Do **not** try to merge them into one class or keep them in lockstep. Their reasons
to change are different: the Pydantic model changes when the pipeline's working
shape changes; the ORM model changes when storage changes. Keeping them separate is
what lets each evolve independently. (Practical note: import them under distinct
names — e.g. `ContentItem` for Pydantic and `ContentItemORM` for SQLAlchemy — so the
one conversion line is unambiguous.)

A subtlety in `model_dump()`: it returns enums and UUIDs as their Python types,
which SQLAlchemy handles. If you later serialize for an API you'll want
`model_dump(mode="json")` instead — but that's the FastAPI plan, not now.

---

## 4. Stage 1 — Ingestion: Hacker News *(confusing step — expanded)*

**File:** `adapters/hn.py` — for V1 this is a plain function `fetch_hn()`, not an
Adapter class. (The Adapter interface gets extracted when Reddit arrives and you
have two fetchers with a shared shape to unify.)

### 4.1 The two-call pattern (the part that confuses people)

HN's API does not return stories directly. It returns **IDs**, and you fetch each
story individually:

1. `GET https://hacker-news.firebaseio.com/v0/topstories.json`
   → a JSON array of up to ~500 integer item IDs, ranked. **No story content.**
2. For each ID: `GET https://hacker-news.firebaseio.com/v0/item/{id}.json`
   → the actual story object.

So a single "fetch" is 1 + N HTTP calls. With N in the hundreds, doing these
sequentially is painfully slow (each round-trip is ~100–300 ms). This is the reason
ingestion uses concurrency.

### 4.2 Concurrency for the per-item fan-out

Use `httpx.AsyncClient` and fire the item requests concurrently with
`asyncio.gather`, bounded by a semaphore so you don't open 500 sockets at once:

```python
import asyncio, httpx

BASE = "https://hacker-news.firebaseio.com/v0"

async def _fetch_item(client, sem, item_id):
    async with sem:                       # cap concurrency (e.g. 20 in flight)
        r = await client.get(f"{BASE}/item/{item_id}.json")
        r.raise_for_status()
        return r.json()

async def fetch_hn(limit: int = 100) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        ids = (await client.get(f"{BASE}/topstories.json")).json()[:limit]
        sem = asyncio.Semaphore(20)
        items = await asyncio.gather(
            *[_fetch_item(client, sem, i) for i in ids],
            return_exceptions=True,       # one dead item shouldn't kill the batch
        )
    return [it for it in items if isinstance(it, dict)]   # drop failures
```

Notes that matter:

- **`limit`**: top-500 is more than V1 needs. Start with ~100; it's the highest-
  ranked, most relevant slice and keeps runs fast and cheap while you iterate.
- **`return_exceptions=True`**: HN occasionally returns `null` or a deleted item.
  Collect what succeeded rather than letting one failure abort the run; the dropped
  count becomes a metric (see §13).
- **The lookback cutoff** (`fetch_lookback_hours`): HN items carry `time` as a Unix
  timestamp. Filter to items newer than `now - fetch_lookback_hours` either here or
  in normalization. For V1, top-stories are already recent, so this is a light
  guard rather than a hard requirement.

### 4.3 Shape of an HN item

A story object looks roughly like:

```json
{
  "id": 12345, "type": "story", "by": "author",
  "time": 1718600000, "title": "…", "url": "https://…",
  "score": 250, "descendants": 80, "kids": [ ... ]
}
```

Two gotchas: **`url` is missing** for "Ask HN"/text posts (it's a self-post), and
**`text` is usually absent** for link posts. Your normalizer must tolerate both.

---

## 5. Stage 2 — Normalization

**File:** `pipeline/normalize.py` — `normalize_hn(raw: dict) -> ContentItem`.

Convert each raw HN dict into a Pydantic `ContentItem`. After this stage, **nothing
downstream knows or cares that the data came from HN** — that's the whole point of
normalization. HN-specific fields that don't fit the common shape go into
`source_metadata` (the JSONB column), not into top-level fields.

Mapping:

| `ContentItem` field | From HN              | Notes                                            |
|---------------------|----------------------|--------------------------------------------------|
| `source`            | `Source.HN`          | constant                                         |
| `source_id`         | `str(raw["id"])`     | string; pairs with `source` for the unique key   |
| `title`             | `raw["title"]`       |                                                  |
| `url`               | `raw.get("url")`     | may be `None` (Ask HN)                            |
| `body`              | `raw.get("text", "")`| often empty for link posts                       |
| `author`            | `raw.get("by")`      |                                                  |
| `published_at`      | from `raw["time"]`   | `datetime.fromtimestamp(t, tz=timezone.utc)`     |
| `source_metadata`   | `{score, descendants}`| HN-specific engagement signals                  |

Malformed items (missing `title`, etc.) are skipped and counted. `items_normalized`
should equal `items_fetched` minus malformed/dropped.

---

## 6. Stage 3 — Deduplication (URL match)

**File:** `pipeline/deduplicate.py` — `deduplicate(items: list[ContentItem]) -> list[ContentItem]`.

The deliberately dumb V1 version: if two items share the same `url`, keep the one
with higher engagement (HN `score` from `source_metadata`) and drop the other.
Items with `url is None` (Ask HN posts) are never treated as duplicates of each
other — they have no shared key.

Algorithm: group by `url`, within each group keep `max(score)`, pass through all
`None`-url items untouched. Output count becomes `items_after_dedup`.

> Fuzzy/embedding-based dedup (cosine > 0.95 across different URLs covering the same
> story) is a §1 non-goal. URL match catches the common case and is trivial to
> reason about.

---

## 7. Stage 4 — Embedding *(confusing step — expanded)*

**File:** `pipeline/embed.py`.

### 7.1 What text actually gets embedded

You embed a **single string per item**, and the obvious-but-wrong choice is "just
the title." Title alone is short and loses context. The V1 choice is
**`title + "\n" + body`** (body often empty for HN, so this gracefully degrades to
the title). The embedding represents the *meaning* of the item; give it the most
meaningful text you have.

### 7.2 Load the model once, encode in one batch

Two performance traps to avoid:

- **Don't reload the model per item.** `SentenceTransformer(model_name)` loads
  weights from disk — do it once at module load (or pass the loaded model in).
- **Don't loop `model.encode(text)` per item.** Pass the whole list to one
  `encode([...])` call. Sentence-transformers batches internally and is far faster
  than N separate calls.

```python
from sentence_transformers import SentenceTransformer
from content_radar.config import settings

_model = SentenceTransformer(settings.embedding_model)   # loaded ONCE

def embed_items(items):
    texts = [f"{it.title}\n{it.body or ''}" for it in items]
    vectors = _model.encode(texts, normalize_embeddings=True)   # see §8.2
    for it, vec in zip(items, vectors):
        it.embedding = vec.tolist()    # store as list[float] for pgvector / Pydantic
    return items
```

### 7.3 Embed the interest profile with the SAME model

The ranking in Stage 5 compares item vectors to the **interest-profile vector**.
Those vectors are only comparable if produced by the *same* embedding model. So the
profile must be embedded with `settings.embedding_model` too — never a different
one. (This is handled by the seeding script in §13/§14; the profile's stored
`embedding` is loaded at rank time.)

### 7.4 Dimension assertion

After encoding, assert `len(items[0].embedding) == EMBEDDING_DIM`. A mismatch here
means model and column disagree — catch it loudly in the pipeline rather than as a
cryptic insert error at save time.

---

## 8. Stage 5 — Ranking: cosine similarity in NumPy *(confusing step — expanded)*

**File:** `pipeline/rank.py`.

### 8.1 Why NumPy in Python (not a pgvector SQL query) for V1

pgvector *can* rank via `ORDER BY embedding <-> target LIMIT k` in SQL, and you'll
move to that when the dataset is large. For V1 you rank in Python because: (a) at a
few hundred items it's instant, (b) you can see and debug the math, and (c) the
items aren't saved yet at ranking time — they're still in-memory objects. Ranking in
Python keeps the whole pipeline a straight line of object transformations.

### 8.2 The cosine subtlety (the part that confuses people)

Cosine similarity between vectors **a** and **b** is:

```
cos(a, b) = (a · b) / (||a|| * ||b||)
```

The trap: a raw dot product is **not** cosine similarity unless the vectors are
**L2-normalized** (unit length). If you normalize every vector to length 1, then
`||a|| = ||b|| = 1`, and cosine **reduces to a plain dot product**. That's why §7.2
passes `normalize_embeddings=True` to `encode()` — it makes the ranking math a
single matrix multiply, with no per-vector division and no chance of forgetting the
normalization.

### 8.3 The computation

With normalized vectors, one query vector vs the item matrix is a single dot product:

```python
import numpy as np

def rank_items(items, profile_embedding):
    profile = np.asarray(profile_embedding, dtype=np.float32)   # shape (384,)
    matrix  = np.asarray([it.embedding for it in items], dtype=np.float32)  # (N, 384)
    scores  = matrix @ profile          # (N,) cosine scores, since all are unit-length
    for it, s in zip(items, scores):
        it.relevance_score = float(s)
    items.sort(key=lambda it: it.relevance_score, reverse=True)   # highest first
    return items
```

If you did **not** normalize at embed time, divide here by the norms — but
normalizing once at the source is cleaner and is the V1 approach.

### 8.4 Sanity metric

Log the score distribution (max, median, min). If the top scores aren't meaningfully
higher than the tail, either the interest profile is too vague or something upstream
is wrong — investigate before trusting the routing.

---

## 9. Stage 6 — Routing (tier assignment)

**File:** `pipeline/route.py`.

Pure list slicing on the already-sorted items, using the config counts:

- items `[0 : deep_summary_count)` → `SummaryTier.DEEP`
- items `[deep_summary_count : deep_summary_count + brief_summary_count)` → `BRIEF`
- everything else → `NONE`

```python
def route_items(sorted_items, deep_n, brief_n):
    for i, it in enumerate(sorted_items):
        if i < deep_n:
            it.summary_tier = SummaryTier.DEEP
        elif i < deep_n + brief_n:
            it.summary_tier = SummaryTier.BRIEF
        else:
            it.summary_tier = SummaryTier.NONE
    return sorted_items
```

This is the cost funnel: only `deep` + `brief` items (e.g. 10 + 20 = 30) ever reach
the LLM; the rest are stored for browsing with no token spend. **Routing quality is
entirely downstream of ranking quality** — if ranking is bad, you pay to summarize
irrelevant items. This is the reason §8's checkpoint comes before turning on §10.

---

## 10. Stage 7 — Summarization *(confusing step — expanded)*

**File:** `pipeline/summarize.py`. Provider-agnostic via the OpenAI-compatible SDK.

### 10.1 One client, either provider

```python
from openai import OpenAI
from content_radar.config import settings

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
```

Pointed at Ollama or DeepSeek depending on config — identical code. Both return the
same response shape, including a `usage` object with `prompt_tokens` and
`completion_tokens`.

### 10.2 Two prompt templates

- **Deep** (top items): explain why this is relevant to *you* specifically, extract
  key takeaways, note anything actionable, focus on practical implications.
- **Brief** (mid items): one sentence capturing the gist and why it might matter.

Keep the prompts as named string templates in `prompts/deep.py` and
`prompts/brief.py` (or constants in `summarize.py` for V1) so you can iterate on
wording without touching pipeline logic. Expect to refine these over days — prompt
quality is the main lever on output quality.

### 10.3 One call per deep/brief item, capturing tokens and cost

```python
def summarize_one(item) -> tuple[str, int]:
    template = DEEP_PROMPT if item.summary_tier == SummaryTier.DEEP else BRIEF_PROMPT
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": template.format(
            title=item.title, body=item.body, url=item.url or "")}],
        temperature=0.3,
    )
    item.summary = resp.choices[0].message.content
    tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
    return item.summary, tokens
```

Accumulate `tokens` across all calls into `total_tokens_used`. Compute cost from a
small price table (per 1M tokens). **Ollama is local, so its cost is `$0`** — tokens
are still tracked, but the dollar figure is zero. DeepSeek V4 Flash bills roughly
`$0.14` per 1M input and `$0.28` per 1M output tokens (verify current rates before
relying on the number):

```python
# USD per 1M tokens; extend as you add models. Local models cost nothing.
PRICING = {
    "llama3.2":          (0.0,  0.0),
    "deepseek-v4-flash": (0.14, 0.28),
}
```

### 10.4 The idempotency / cost guard (do not skip this)

Re-running the pipeline must **not** re-summarize and re-bill items already done.
Before the LLM loop, look up which `(source, source_id)` pairs already have a stored
summary, and skip those:

```python
already = content_repo_existing_summary_ids()   # {(source, source_id), ...} from DB
to_summarize = [it for it in items
                if it.summary_tier != SummaryTier.NONE
                and (it.source, it.source_id) not in already]
```

This guard is what makes Stage 7 safe to re-run. Without it, every run pays again for
the same content — the single most expensive bug you could ship in this pipeline.

### 10.5 Failure handling

Wrap each call so one failed summary doesn't abort the run: on error, leave
`summary = None`, increment a `summarization_failures` counter, and continue. A
missing summary is recoverable on the next run; a crashed run mid-way is not.

---

## 11. Stage 8 — Persistence: idempotent upsert + run lifecycle *(confusing step — expanded)*

**File:** `db/persist.py` — plain functions (no Repository class yet).

### 11.1 Idempotent upsert on `(source, source_id)`

Your `content_items` table has `UniqueConstraint("source", "source_id")`. Use it
with Postgres `INSERT ... ON CONFLICT ... DO UPDATE` so a re-run **updates** existing
rows (e.g. a new score or a freshly added summary) instead of inserting duplicates:

```python
from sqlalchemy.dialects.postgresql import insert
from content_radar.db.models import ContentItem as ContentItemORM

def upsert_items(session, items, batch_id):
    rows = []
    for it in items:
        d = it.model_dump()           # Pydantic -> dict (the §3 boundary)
        d["batch_id"] = batch_id
        rows.append(d)

    stmt = insert(ContentItemORM).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_source_item",                 # the named unique constraint
        set_={
            "relevance_score": stmt.excluded.relevance_score,
            "summary_tier":    stmt.excluded.summary_tier,
            "summary":         stmt.excluded.summary,
            "embedding":       stmt.excluded.embedding,
            "batch_id":        stmt.excluded.batch_id,
        },
    )
    session.execute(stmt)
```

`stmt.excluded` refers to the row that *would have* been inserted — i.e. the new
values. You update only the fields a re-run should refresh, leaving identity and
ingest timestamps alone.

### 11.2 The `PipelineRun` lifecycle (parent row + metrics)

The run row is created **first**, so every content item can reference it via
`batch_id`, and **finalized last** with the accumulated metrics:

1. **Start:** insert a `PipelineRun(status=RUNNING)`, `flush()` to get its `id`.
2. **Run stages**, accumulating counters/tokens/cost in memory.
3. **Save items** with `batch_id = run.id` (§11.1).
4. **Finish (success):** set `completed_at`, `status=COMPLETED`, and write
   `items_ingested`, `items_after_dedup`, `items_summarized`, `total_tokens_used`,
   `total_cost_usd`, `embedding_model`, `summary_model`.
5. **Finish (failure):** set `status=FAILED` and `error_log`, then re-raise.

This is what populates your monitoring data later — every run leaves an auditable
record of what it processed and what it cost.

---

## 12. Orchestration — `run_pipeline.py`

**File:** `run_pipeline.py` (top level). The entire pipeline as one readable,
top-to-bottom function. This is **intentionally not** an orchestrator class or a
stage registry — a linear script is the correct design at this size, and the
orchestrator pattern gets extracted around the monitoring iteration, when there are
enough stages and metrics that flat code genuinely hurts.

```python
def main():
    run = create_pipeline_run()                       # §11.2 step 1 (RUNNING)
    try:
        raw     = asyncio.run(fetch_hn(limit=100))    # Stage 1
        items   = [normalize_hn(r) for r in raw]      # Stage 2
        items   = [i for i in items if i]             #   drop malformed
        items   = deduplicate(items)                  # Stage 3
        items   = embed_items(items)                  # Stage 4
        profile = load_interest_profile_embedding()   #   seeded row (§13)
        items   = rank_items(items, profile)          # Stage 5
        items   = route_items(items, settings.deep_summary_count,
                                     settings.brief_summary_count)  # Stage 6
        items, metrics = summarize(items)             # Stage 7 (with §10.4 guard)
        with get_session() as session:                # Stage 8
            upsert_items(session, items, run.id)
        finalize_run(run, items, metrics, status="completed")   # §11.2 step 4
    except Exception as exc:
        finalize_run(run, error=str(exc), status="failed")      # §11.2 step 5
        raise

if __name__ == "__main__":
    main()
```

You can read the entire data flow in one screen — that legibility is the V1 goal.

---

## 13. Metrics & Verification

### 13.1 Quantitative metrics (the `pipeline_runs` row)

Every metric V1 tracks is already a column on `pipeline_runs`, collected by the run
script and written once at the end:

| Metric                  | Source                                   |
|-------------------------|------------------------------------------|
| `items_ingested`        | count after Stage 1 (post-failure drop)  |
| `items_after_dedup`     | count after Stage 3                       |
| `items_summarized`      | successful LLM calls in Stage 7           |
| `total_tokens_used`     | summed `usage` across Stage 7 calls       |
| `total_cost_usd`        | tokens × `PRICING` (0 for Ollama)         |
| `embedding_model`       | `settings.embedding_model`                |
| `summary_model`         | `settings.llm_model`                      |
| `started/completed_at`  | run lifecycle (latency = the difference)  |
| `status`, `error_log`   | success/failure                           |

Two ephemeral counters worth logging (not yet columns): HN items dropped during
fetch, and summarization failures.

### 13.2 The one quality metric: manual precision@k

V1 deliberately uses **human judgment** for ranking quality, not automated eval.
After a run, read the top 10 `deep` items in Supabase and ask: do these actually
match my interests? That's precision@k by eye. Automated evaluation (LLM-as-judge,
precision@k against labeled data, retrieval metrics) is deferred because you have no
feedback/label data yet to evaluate against — that arrives with the feedback loop in
a later iteration. Trying to automate eval now would be building a measuring
instrument before there's anything to measure.

---

## 14. Build Order & Checkpoints

Build and verify in this order. The two checkpoints are the whole point — they let
you trust each half before building the next.

1. **Seed the interest profile.** Write `scripts/seed_interest_profile.py`: embed
   your interest paragraph with `settings.embedding_model` and insert one
   `interest_profile` row. (Do this early — ranking can't run without it.)
2. **Stage 1 — `adapters/hn.py`.** Run it standalone; print a few raw items.
3. **Stage 2 — `pipeline/normalize.py`.** Confirm raw dicts become valid
   `ContentItem`s and HN fields land in `source_metadata`.
4. **Stage 3 — `pipeline/deduplicate.py`.** Confirm same-URL collisions collapse.
5. **Stage 4 — `pipeline/embed.py`.** Confirm vectors are length 384 and normalized.
6. **Stage 5 — `pipeline/rank.py`.** Print the sorted titles + scores.
7. **Stage 8 (save) wired early, summarization still OFF.** Run ingest→rank→save.
   - **✅ Checkpoint 1:** open Supabase, read `content_items` ordered by
     `relevance_score`. Do the top items match your interests? Fix ranking/profile
     here, *before* spending any tokens.
8. **Stage 6 — `pipeline/route.py`.** Confirm tier counts (e.g. 10 / 20 / rest).
9. **Stage 7 — `pipeline/summarize.py`.** Turn on the LLM. Start with Ollama (free)
   to debug prompts at zero cost; switch `llm_model`/`llm_base_url` to DeepSeek once
   prompts are good.
10. **`run_pipeline.py`.** Wire all stages + the run lifecycle into the linear script.
    - **✅ Checkpoint 2:** run it **twice**. Confirm the second run does **not**
      duplicate rows (upsert works) and does **not** re-bill already-summarized items
      (the §10.4 guard works). Confirm a `pipeline_runs` row exists with sane
      counters and cost.

When both checkpoints pass, V1 is done: a single command pulls HN, ranks it against
your interests, summarizes the top slice, and stores everything idempotently with a
full metrics record — and you've proven it's safe to re-run.