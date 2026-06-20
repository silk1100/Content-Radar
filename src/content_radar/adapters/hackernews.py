"""Hacker News adapter.

Two-phase fetch, both bounded and concurrent:
  1. story metadata — HN returns IDs, then one call per item (the 1 + N pattern)
  2. article text   — for stories that have a `url`, fetch the linked page and
                      extract its body, attaching it to the story dict in place
                      as `article_text`

Design rules this file follows:
  - HTTP is I/O  -> async, concurrent, bounded by a semaphore
  - HTML parsing -> CPU, kept in utils.py, run via asyncio.to_thread
  - one failure (dead item, unreachable page) never aborts the batch
  - article text is attached to each story object, so there are no parallel
    index-aligned lists to keep in sync
"""

import asyncio
from typing import Any, List, Literal

import httpx

from .utils import extract_page_content

URLS = {
    "new": "https://hacker-news.firebaseio.com/v0/newstories.json",
    "top": "https://hacker-news.firebaseio.com/v0/topstories.json",
    "best": "https://hacker-news.firebaseio.com/v0/beststories.json",
}

STORY_URL = "https://hacker-news.firebaseio.com/v0/item/{story_id}.json"

StoryType = Literal["new", "top", "best", "all"]

# HN's Firebase API is fast and reliable, so a generous fan-out is fine there.
# Article fetches hit the open web (slow, flaky), so they get a tighter cap.
STORY_CONCURRENCY = 20
ARTICLE_CONCURRENCY = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,...",
    "Accept-Language": "en-US,en;q=0.9",
}

class HackerNewsAdapter:
    def __init__(self, content_type: StoryType | None = None, limit: int = 10):
        self.content_type = content_type or "all"
        self.limit = limit

        valid_content_types = {"new", "top", "best", "all"}
        if self.content_type not in valid_content_types:
            raise ValueError(
                f"Invalid content_type={self.content_type}. "
                f"Expected one of {valid_content_types}"
            )

    # --- low-level fetch -----------------------------------------------------

    async def _fetch_data(
        self,
        client: httpx.AsyncClient,
        url: str | None,
        return_type: str = "json",
    ) -> Any:
        """GET a URL. Returns parsed JSON or raw text; None on any HTTP error
        so callers can drop the failure rather than crash the batch."""
        if not url:
            return None
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.text if return_type == "text" else response.json()
        except httpx.HTTPError as error:
            print(f"Failed to fetch {url}: {error}")
            return None

    async def _fetch_story(
        self, client: httpx.AsyncClient, sem: asyncio.Semaphore, story_id: int
    ) -> dict | None:
        async with sem:
            story = await self._fetch_data(
                client, STORY_URL.format(story_id=story_id)
            )
        return story if isinstance(story, dict) else None

    async def _fetch_stories_by_type(
        self, client: httpx.AsyncClient, content_type: str
    ) -> List[dict]:
        story_ids = await self._fetch_data(client, URLS[content_type])
        if not isinstance(story_ids, list):
            return []

        story_ids = story_ids[: self.limit]
        sem = asyncio.Semaphore(STORY_CONCURRENCY)
        stories = await asyncio.gather(
            *(self._fetch_story(client, sem, sid) for sid in story_ids),
            return_exceptions=True,
        )
        return [s for s in stories if isinstance(s, dict)]

    # --- article text --------------------------------------------------------

    async def _attach_article_text(
        self, client: httpx.AsyncClient, sem: asyncio.Semaphore, story: dict
    ) -> None:
        """Fetch the story's linked page, extract its body, attach in place as
        `article_text`. No-op for Ask HN / text posts (no url). Extraction is
        CPU-bound, so it runs in a thread to avoid blocking the event loop."""
        url = story.get("url")
        if not url:
            return

        async with sem:
            html = await self._fetch_data(client, url, return_type="text")

        story["article_text"] = await asyncio.to_thread(
            extract_page_content, html, story.get("title", "")
        )

    async def _attach_article_texts(
        self, client: httpx.AsyncClient, stories: List[dict]
    ) -> List[dict]:
        sem = asyncio.Semaphore(ARTICLE_CONCURRENCY)
        await asyncio.gather(
            *(self._attach_article_text(client, sem, s) for s in stories),
            return_exceptions=True,  # one bad page must not abort the batch
        )
        return stories

    # --- public entry point --------------------------------------------------

    async def fetch_data(self) -> List[dict]:
        # follow_redirects matters: many article URLs 301/302, and httpx does
        # NOT follow by default — without this you'd get redirect stubs as HTML.
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=HEADERS) as client:
            if self.content_type == "all":
                results = await asyncio.gather(
                    *(self._fetch_stories_by_type(client, ct) for ct in URLS)
                )
                stories = [s for group in results for s in group]
            else:
                stories = await self._fetch_stories_by_type(
                    client, self.content_type
                )

            await self._attach_article_texts(client, stories)
            return stories


if __name__ == "__main__":
    adapter = HackerNewsAdapter(content_type="top", limit=5)
    results = asyncio.run(adapter.fetch_data())

    for story in results:
        print(story.get("title"))
        print(story.get("url"))
        print("--- article_text (first 300 chars) ---")
        print((story.get("article_text") or "")[:1000])
        print("=======================================")
    x=0