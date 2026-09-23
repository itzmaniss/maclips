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
    additions = {
        "sources": {"state_json": "TEXT NOT NULL DEFAULT '{}'"},
        "clips": {"candidate_key": "TEXT", "data_json": "TEXT NOT NULL DEFAULT '{}'"},
        "posts": {"disclosure_ticked": "INTEGER NOT NULL DEFAULT 0", "bundle_path": "TEXT"},
    }
    for table, columns in additions.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for column, declaration in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clip_candidate ON clips(source_id,candidate_key)")
    conn.commit()
    return conn


def log_event(conn, event, source_id=None, clip_id=None, campaign_id=None):
    import time
    conn.execute("INSERT INTO time_log(event,at,source_id,clip_id,campaign_id) VALUES(?,?,?,?,?)",
                 (event, time.time(), source_id, clip_id, campaign_id))


def register_source(conn, content_hash, path, clip_class, campaign_id=None, duration_s=None, state=None):
    import json
    row = conn.execute("SELECT * FROM sources WHERE content_hash=?", (content_hash,)).fetchone()
    if row and (row["clip_class"] != clip_class or row["campaign_id"] != campaign_id):
        raise ValueError("source already belongs to a different class/campaign; do not silently reroute it")
    if row:
        if state is not None:
            conn.execute("UPDATE sources SET state_json=? WHERE id=?", (json.dumps(state), row["id"]))
        return row["id"]
    cur = conn.execute("INSERT INTO sources(content_hash,path,clip_class,campaign_id,duration_s,state_json) VALUES(?,?,?,?,?,?)",
                       (content_hash,str(path),clip_class,campaign_id,duration_s,json.dumps(state or {})))
    log_event(conn,"source_registered",cur.lastrowid,campaign_id=campaign_id)
    return cur.lastrowid


def save_candidate(conn, source_id, clip_class, candidate, previews=None):
    import json
    from .tracking import candidate_key
    key = candidate_key(candidate)
    row = conn.execute("SELECT id,data_json FROM clips WHERE source_id=? AND candidate_key=?",(source_id,key)).fetchone()
    if row:
        # Replaying a cache must not reset human edits or decisions.
        data=json.loads(row["data_json"])
        if previews is not None:
            data["previews"]=previews
            conn.execute("UPDATE clips SET data_json=? WHERE id=?",(json.dumps(data),row["id"]))
        return row["id"]
    data={**candidate,"previews":previews or {},"commentary_mode":"none"}
    cur=conn.execute("INSERT INTO clips(source_id,candidate_key,start_word,end_word,start_s,end_s,hook_text,clip_class,data_json) VALUES(?,?,?,?,?,?,?,?,?)",
                     (source_id,key,candidate.get("start_word"),candidate.get("end_word"),candidate["start"],candidate["end"],candidate.get("hook_text",""),clip_class,json.dumps(data)))
    return cur.lastrowid


def record_post(conn, clip_id, platform, account, row):
    from urllib.parse import urlparse
    from .compliance import validate_post
    clip=conn.execute("SELECT * FROM clips WHERE id=?",(clip_id,)).fetchone()
    if not clip:
        raise ValueError("unknown clip")
    validate_post(clip["clip_class"],row)
    if clip["review_decision"] != "approved" or not clip["render_path"]:
        raise ValueError("post tracking requires an approved rendered clip")
    url=str(row.get("post_url", "")).strip()
    parsed=urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("a complete HTTPS post URL is required")
    import time
    status=row.get("status","posted")
    if status not in {"posted","submitted","approved","rejected"}:
        raise ValueError("unknown post status")
    old=conn.execute("SELECT * FROM posts WHERE clip_id=? AND platform=? AND account=?",(clip_id,platform,account)).fetchone()
    now=time.time()
    posted=old["posted_at"] if old and old["posted_at"] else now
    submitted=(old["submitted_at"] if old else None) or (now if status in {"submitted","approved"} else None)
    values=(url,posted,submitted,status,row.get("rejection_reason"),row.get("views"),int(row.get("disclosure_ticked") is True))
    if old:
        conn.execute("UPDATE posts SET post_url=?,posted_at=?,submitted_at=?,status=?,rejection_reason=?,views=?,disclosure_ticked=? WHERE id=?",(*values,old["id"]))
        pid=old["id"]
    else:
        cur=conn.execute("INSERT INTO posts(clip_id,platform,account,post_url,posted_at,submitted_at,status,rejection_reason,views,disclosure_ticked) VALUES(?,?,?,?,?,?,?,?,?,?)",(clip_id,platform,account,*values))
        pid=cur.lastrowid
    if not old or old["post_url"] != url:
        log_event(conn,"post_url_pasted",clip["source_id"],clip_id)
    if status in {"submitted","approved"} and (not old or not old["submitted_at"]):
        log_event(conn,"post_submitted",clip["source_id"],clip_id)
    return pid
