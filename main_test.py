import asyncio
from collections import Counter

from src.content_radar.config import settings
from src.content_radar.adapters.hackernews import HackerNewsAdapter
from src.content_radar.pipeline.normalize import normalize_hn_batch
from src.content_radar.pipeline.deduplicate import deduplicate
from src.content_radar.pipeline.embed import embed_items
from src.content_radar.pipeline.rank import rank_items, load_interest_profile_embedding
from src.content_radar.pipeline.route import route_items
from src.content_radar.pipeline.summarize import summarize
from src.content_radar.models.schemas import SummaryTier, PipelineStatus
from src.content_radar.db.client import get_session
from src.content_radar.db.persist import (
    create_pipeline_run, upsert_items, finalize_run, existing_summary_keys,
)


def build_routed_items():
    adapter = HackerNewsAdapter(content_type="top", limit=100)
    raw = asyncio.run(adapter.fetch_data())
    normalized = normalize_hn_batch(raw)
    deduped = deduplicate(normalized)
    embedded = embed_items(deduped)
    profile = load_interest_profile_embedding()
    ranked = rank_items(embedded, profile)
    routed = route_items(ranked)            # assign tiers
    return routed, len(normalized), len(deduped)


def save(items, ingested, after_dedup, metrics):
    with get_session() as session:
        run = create_pipeline_run(
            session,
            embedding_model=settings.embedding_model,
            summary_model=settings.llm_model,
        )
        upsert_items(session, items, run.id)
        finalize_run(
            session, run,
            status=PipelineStatus.COMPLETED,
            items_ingested=ingested,
            items_after_dedup=after_dedup,
            items_summarized=metrics.items_summarized,
            total_tokens_used=metrics.total_tokens,
            total_cost_usd=metrics.total_cost_usd,
        )


def main():
    items, ingested, after_dedup = build_routed_items()

    tiers = Counter(it.summary_tier.value for it in items)
    print(f"tiers -> deep={tiers['deep']} brief={tiers['brief']} none={tiers['none']}")
    print("=" * 60)

    # cost guard input: what's already summarized in the DB?
    with get_session() as s:
        skip = existing_summary_keys(s)
    print(f"already-summarized in DB (will be skipped): {len(skip)}")

    # FIRST summarize
    items, metrics = summarize(items, skip_keys=frozenset(skip))
    print(f"summarized={metrics.items_summarized} skipped={metrics.skipped} "
          f"failed={metrics.failures} tokens={metrics.total_tokens} "
          f"cost=${metrics.total_cost_usd:.4f}")
    if metrics.failures and metrics.items_summarized == 0:
        print("  (all failed — is Ollama running? `ollama serve` + `ollama pull llama3.2`)")
    print("=" * 60)

    # eyeball a couple of DEEP summaries
    deep = [it for it in items if it.summary_tier == SummaryTier.DEEP and it.summary]
    for it in deep[:2]:
        print(f"DEEP: {it.title[:60]}")
        print(f"  {it.summary}")
        print("-" * 60)

    # persist (now with summaries)
    save(items, ingested, after_dedup, metrics)

    # COST-GUARD PROOF: re-summarize the SAME items; should do zero LLM work
    with get_session() as s:
        skip2 = existing_summary_keys(s)
    _, metrics2 = summarize(items, skip_keys=frozenset(skip2))
    print(f"second pass: summarized={metrics2.items_summarized} (should be 0), "
          f"skipped={metrics2.skipped}")
    print(f"cost guard works (no re-billing)? "
          f"{'OK' if metrics2.items_summarized == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()