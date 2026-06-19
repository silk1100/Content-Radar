"""Database connectivity — the engine and the session factory.

Everything that touches Postgres goes through this module. It defines two things:

    engine        -> the connection pool; created ONCE for the whole process
    SessionLocal  -> a factory that hands out Session objects (units of work)

Note what's NOT here: any knowledge of your tables. This file only knows how to
reach the database. Table definitions live in models.py. The separation means
'how do I connect' and 'what is stored' can change independently.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings

# The Engine is the single source of connectivity for the entire process. It is a
# factory + connection pool, not a live connection. Create one, reuse it forever.
engine = create_engine(
    settings.SUPABASE_DATABASE_URL,
    # Hosted Postgres and the Supavisor pooler silently drop idle connections; without
    # this you get intermittent "server closed the connection unexpectedly" errors that
    # are maddening to debug. It's cheap insurance — always on for hosted databases.
    pool_pre_ping=True,
    # echo=True logs every SQL statement SQLAlchemy generates. Leave it on while you're
    # learning — seeing the actual INSERT/SELECT it emits is the fastest way to build a
    # mental model of what the ORM does. Wire it to a config flag and silence it later.
    echo=True,
)
# sessionmaker is a factory pre-bound to our engine. SessionLocal() returns a fresh
# Session. expire_on_commit=False keeps your objects usable after commit instead of
# invalidating their attributes — convenient for a script that commits and then keeps
# reading the same objects it just saved.
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a Session that commits on success and rolls back on any error.

    This wrapper enforces the single rule that prevents most database bugs: every
    unit of work is either fully committed or fully rolled back, and the connection
    is always returned to the pool. You never leak connections and never leave a
    half-written transaction behind.

    Usage:
        with get_session() as session:
            session.add(item)
        # commit + close happen automatically on exit
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()  # nothing was wrong -> persist the whole transaction
    except Exception:
        session.rollback()  # something failed -> undo every pending change
        raise
    finally:
        session.close()  # always hand the connection back to the pool
