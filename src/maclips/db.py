"""SQLite tracking store. stdlib sqlite3, no ORM.

Tables follow PLAN.md §7.3. `time_log` is what feeds the throughput numbers in
§6.4 and, through them, the business gate in §1.5 — so it exists from step 1
even though nothing writes to it yet.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id                 INTEGER PRIMARY KEY,
    brand              TEXT NOT NULL,
    campaign_name      TEXT NOT NULL,
    category           TEXT,
    rate_per_1k        REAL,
    platforms_allowed  TEXT,
    min_duration_s     REAL,
    max_duration_s     REAL,
    config_json        TEXT NOT NULL,
    raw_brief          TEXT NOT NULL,
    confirmed_at       REAL,
    created_at         REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS sources (
    id             INTEGER PRIMARY KEY,
    content_hash   TEXT NOT NULL UNIQUE,
    path           TEXT NOT NULL,
    clip_class     TEXT NOT NULL CHECK (clip_class IN ('campaign', 'general-own')),
    campaign_id    INTEGER REFERENCES campaigns(id),
    duration_s     REAL,
    retention_date TEXT,
    created_at     REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS clips (
    id              INTEGER PRIMARY KEY,
    source_id       INTEGER NOT NULL REFERENCES sources(id),
    start_word      INTEGER,
    end_word        INTEGER,
    start_s         REAL,
    end_s           REAL,
    layout          TEXT,
    hook_text       TEXT,
    commentary      TEXT,
    clip_class      TEXT NOT NULL,
    render_path     TEXT,
    review_decision TEXT,
    review_reason   TEXT,
    created_at      REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS posts (
    id               INTEGER PRIMARY KEY,
    clip_id          INTEGER NOT NULL REFERENCES clips(id),
    platform         TEXT NOT NULL,
    account          TEXT NOT NULL,
    post_url         TEXT,
    posted_at        REAL,
    submitted_at     REAL,
    status           TEXT,
    rejection_reason TEXT,
    views            INTEGER
);

CREATE TABLE IF NOT EXISTS time_log (
    id          INTEGER PRIMARY KEY,
    event       TEXT NOT NULL,
    at          REAL NOT NULL DEFAULT (unixepoch()),
    source_id   INTEGER REFERENCES sources(id),
    clip_id     INTEGER REFERENCES clips(id),
    campaign_id INTEGER REFERENCES campaigns(id)
);

CREATE INDEX IF NOT EXISTS idx_clips_source ON clips(source_id);
CREATE INDEX IF NOT EXISTS idx_posts_clip   ON posts(clip_id);
CREATE INDEX IF NOT EXISTS idx_timelog_src  ON time_log(source_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the store, creating it if absent. WAL so the web UI can read mid-run."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn
