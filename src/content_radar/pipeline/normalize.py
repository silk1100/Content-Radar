from ..models.schemas import ContentItem, Source
from typing import List
import datetime as dt
import asyncio



class Normalize:
    def __init__(self):
        pass

    def _normalize_hn_story(self, story: dict) -> ContentItem:
        return ContentItem(
            source=Source.HN,
            source_id=story.get("id", 0),
            title=story.get("title", "no-title"),
            author=story.get("by", "no-author"),
            published_at=dt.datetime.fromtimestamp(story.get("time", 0)),
            body=story.get("title", "no-body"),
        )

    async def _normalize_hn_story_async(self, story: dict) -> ContentItem:
        return self._normalize_hn_story(story)

    async def normalize_hn(self, stories: List[dict]) -> List[ContentItem]:
        """
        {'by': 'giuliomagnifico', 'descendants': 367, 'id': 48565498,
        'kids': [48578257, 48578183, 48573764],
        'score': 340,
        'text': '<a href="https:&#x2F;&#x2F;archive.ph&#x2F;MlU1U" rel="nofollow">https:&#x2F;&#x2F;archive.ph&#x2F;MlU1U</a>',
        'time': 1781668516,
        'title': 'US holds off\xa0blacklisting DeepSeek, more than 100 firms deemed security risks', 'type': 'story',
        'url': 'https://www.reuters.com/world/china/us-holds-off-blacklisting-chinas-deepseek-more-than-100-firms-deemed-security-2026-06-17/'}
        """
        # Write async code to call self._normalize_hn_story
        valid_stories = [
           story 
           for story in stories
           if story.get("type") == "story"
           and not story.get("deleted", False)
           and not story.get("dead", False)
       ]
        tasks = [
           self._normalize_hn_story_async(story)
           for story in valid_stories
        ]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    normalize = Normalize()
    normalize
