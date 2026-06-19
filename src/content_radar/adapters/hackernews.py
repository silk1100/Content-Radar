import asyncio
from typing import Any, List, Literal

import httpx

URLS = {
    "new": "https://hacker-news.firebaseio.com/v0/newstories.json",
    "top": "https://hacker-news.firebaseio.com/v0/topstories.json",
    "best": "https://hacker-news.firebaseio.com/v0/beststories.json",
}

STORY_URL = "https://hacker-news.firebaseio.com/v0/item/{story_id}.json"

StoryType = Literal["new", "top", "best", "all"]


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

    async def _fetch_data(self, client: httpx.AsyncClient, url: str) -> Any:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            print(f"Failed to fetch {url}: {error}")
            return None

    async def _fetch_story(
        self, client: httpx.AsyncClient, story_id: int
    ) -> dict | None:
        url = STORY_URL.format(story_id=story_id)
        story = await self._fetch_data(client, url)

        if not isinstance(story, dict):
            return None

        return story

    async def _fetch_stories_by_type(
        self,
        client: httpx.AsyncClient,
        content_type: str,
    ) -> List[dict]:
        ids_url = URLS[content_type]

        story_ids = await self._fetch_data(client, ids_url)

        if not isinstance(story_ids, list):
            return []

        story_ids = story_ids[: self.limit]

        tasks = [self._fetch_story(client, story_id) for story_id in story_ids]

        stories = await asyncio.gather(*tasks)

        return [story for story in stories if story is not None]

    async def fetch_data(self) -> List[dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if self.content_type == "all":
                tasks = [
                    self._fetch_stories_by_type(client, content_type)
                    for content_type in URLS
                ]

                results = await asyncio.gather(*tasks)

                flattened_results = [story for stories in results for story in stories]

                return flattened_results

            return await self._fetch_stories_by_type(
                client=client,
                content_type=self.content_type,
            )


if __name__ == "__main__":
    adapter = HackerNewsAdapter(content_type="top", limit=5)
    results = asyncio.run(adapter.fetch_data())

    for story in results:
        print(story.get("title"))
        print(story.get("url"))
        print(story)
        print("=======================================")
