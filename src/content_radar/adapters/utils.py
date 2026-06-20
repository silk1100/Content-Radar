"""HTML → article text extraction.

These functions are PURE CPU work (parsing strings), deliberately synchronous.
The adapter does the async HTTP fetching and calls extract_page_content via
asyncio.to_thread so the (blocking) parsing never stalls the event loop.

Extraction strategy is a fallback chain:
  1. trafilatura  — reliable, precision-tuned readability extractor (primary)
  2. BS4 heuristic — locate the element containing the title, return its
                     direct children's text (last-ditch, only if trafilatura
                     returns nothing)

Every function returns "" on any failure, so the caller can always fall back
to the title when building the ContentItem body.
"""

import re

import trafilatura
from bs4 import BeautifulSoup


def normalize_text(text: str) -> str:
    """Collapse whitespace, strip, lowercase — for fuzzy phrase matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def get_first_words(text: str, n: int = 5) -> str:
    """First n normalized words of `text`, used as a search phrase in the DOM."""
    return " ".join(normalize_text(text).split()[:n])


def extract_with_trafilatura(html: str) -> str:
    """Primary extractor. Returns the article body with boilerplate removed.

    favor_precision biases toward dropping uncertain blocks (nav, related links)
    rather than including them — the right trade-off when the downstream embedder
    (MiniLM) only sees ~256 tokens, so every boilerplate token is one fewer
    article token inside the window. Tables are excluded for the same reason:
    on news/blog pages they're usually nav grids, not topic signal.
    """
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return text or ""


def extract_children_with_bs4(
    html: str,
    target_text: str,
    separator: str = "\n\n",
) -> str:
    """Fallback heuristic. Find the first element whose text contains the start
    of the title, then return the joined text of its DIRECT children.

    This is the fragile path (the first match is often a broad wrapper whose
    children include some chrome), so it only runs when trafilatura yields
    nothing. Returns "" on any structural surprise rather than raising.
    """
    search_phrase = get_first_words(target_text, 5)
    if not search_phrase:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup  # some fragments/malformed pages have no <body>

    matched_element = None
    for element in body.find_all(True):
        element_text = normalize_text(element.get_text(" ", strip=True))
        if search_phrase in element_text:
            matched_element = element
            break

    if matched_element is None:
        return ""

    children_texts = [
        child.get_text(" ", strip=True)
        for child in matched_element.find_all(recursive=False)
        if child.get_text(strip=True)
    ]
    return separator.join(children_texts)


def extract_page_content(
    html: str | None,
    target_text: str | None = None,
    separator: str = "\n\n",
) -> str:
    """Turn raw HTML into article text. Pure CPU — call via asyncio.to_thread.

    Note this does NOT fetch: the adapter owns the (async, concurrent) HTTP and
    hands the already-downloaded HTML here. Priority is trafilatura first
    (reliable), BS4 second (heuristic safety net). Returns "" if both fail or
    `html` is None, so the caller falls back to the title.
    """
    if not html:
        return ""

    clean_text = extract_with_trafilatura(html)
    if clean_text:
        return clean_text

    if target_text:
        return extract_children_with_bs4(html, target_text, separator)

    return ""