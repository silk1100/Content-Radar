"""One-off script to create the tables. Run with:

    python -m content_radar.db.init_db

This is deliberately create_all(), NOT migrations (Alembic). create_all only CREATES
tables that are missing — it never ALTERs or DROPs. That's exactly right while the
schema is still molten and you can freely drop-and-recreate. The day you need to change
a column on a table that already holds data you care about is the day Alembic earns its
place. Not before — same 'extract, don't install' rule we've been following.
"""

from .client import engine

# This import looks decorative but it's load-bearing. Importing models.py EXECUTES it,
# which DEFINES every ORM class, which REGISTERS every table on Base.metadata. Skip the
# import and create_all sees nothing. (All your models live in one file today, so this
# single line covers them all. If you ever split models across files, you must import
# each file here, or its table silently won't be created.)
from .models import Base


def create_tables() -> None:
    """Create all registered tables that don't already exist.

    Base.metadata is the catalogue SQLAlchemy assembled as each model class was
    defined. create_all walks it, compares against the live database, and issues a
    CREATE TABLE for anything missing — in dependency order, so pipeline_runs is
    created before content_items, which references it via batch_id.
    """
    Base.metadata.create_all(bind=engine)
    print("Tables created (or already existed).")


if __name__ == "__main__":
    create_tables()
