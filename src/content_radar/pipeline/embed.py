"""Stage 4 — Embedding.

Turn text into a vector that represents its meaning. Two callers go through the
SAME private chokepoint (_encode), which is what guarantees their vectors live
in one comparable space:

  - embed_items(items)  -> the pipeline, for ContentItems
  - embed_query(text)   -> the seed script, for the interest profile

If those two ever diverged in model or normalization, every ranking would be
silently meaningless. Routing both through _encode makes "same space" a
structural guarantee rather than a convention you must remember.

Three things that quietly degrade ranking if you get them wrong:
  1. Load the model ONCE (module level), not per call.
  2. Encode a whole batch in ONE encode([...]) call.
  3. normalize_embeddings=True — unit-length vectors make cosine similarity
     reduce to a plain dot product in Stage 5.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..models.schemas import ContentItem

logger = logging.getLogger(__name__)

# Must match the Vector(EMBEDDING_DIM) column in db/models.py.
EMBEDDING_DIM = 384

# MiniLM truncates at ~256 tokens anyway; this just stops a giant article body
# from being carried into encode() for nothing.
MAX_EMBED_CHARS = 2000

# Loaded ONCE at import, reused for every call.
_model = SentenceTransformer(settings.embedding_model)


def _encode(texts: List[str]) -> np.ndarray:
    """The single chokepoint: text -> L2-normalized vectors.

    Both items and the interest profile pass through here, so their vectors are
    guaranteed comparable. The dimension assert lives here (one place) and turns
    a model/column mismatch into a loud, named failure instead of a cryptic
    insert error three stages later.
    """
    vectors = _model.encode(texts, normalize_embeddings=True)
    dim = vectors.shape[1]
    assert dim == EMBEDDING_DIM, (
        f"Embedding dim {dim} != EMBEDDING_DIM {EMBEDDING_DIM}. "
        f"Model '{settings.embedding_model}' and the Vector() column disagree — "
        f"fix EMBEDDING_DIM and recreate the table, or use a {EMBEDDING_DIM}-dim model."
    )
    return vectors


def _item_text(item: ContentItem) -> str:
    """The string we embed for one item: title + body (body often empty)."""
    text = f"{item.title}\n{item.body or ''}"
    return text[:MAX_EMBED_CHARS]


def embed_items(items: List[ContentItem]) -> List[ContentItem]:
    """Fill each item's `embedding` in place. Returns the same list."""
    if not items:
        return items

    vectors = _encode([_item_text(it) for it in items])
    for it, vec in zip(items, vectors):
        it.embedding = vec.tolist()   # list[float] for pgvector / Pydantic

    logger.info("embedded %d items (dim=%d, normalized)", len(items), vectors.shape[1])
    return items


def embed_query(text: str) -> List[float]:
    """Embed a single query string (e.g. the interest profile) IDENTICALLY to
    how items are embedded, so rank's dot-product is a valid cosine similarity."""
    vector = _encode([text[:MAX_EMBED_CHARS]])[0]
    return vector.tolist()