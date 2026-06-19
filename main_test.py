import asyncio

from src.content_radar.adapters.hackernews import HackerNewsAdapter
from src.content_radar.pipeline.normalize import Normalize

if __name__ == "__main__":
    adapter = HackerNewsAdapter()
    results = asyncio.run(adapter.fetch_data())

    for story in results:
        print(story.get("title"))
        print(story.get("url"))
        print(story)
        print("=======================================")

    normalizer = Normalize()
    normalizer.normalize_hn(results)
    x = 0
