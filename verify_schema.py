#!/usr/bin/env python3
"""Compare SQLite schema to archive CSV headers. Exit 0 if match; else print diff and exit 1."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

EXPECTED: dict[str, list[str]] = {
    "users": [
        "user_id",
        "user_gender",
        "user_age",
        "age_group",
        "country",
        "location",
        "interests",
    ],
    "campaigns": [
        "campaign_id",
        "name",
        "start_date",
        "end_date",
        "duration_days",
        "total_budget",
    ],
    "ads": [
        "ad_id",
        "campaign_id",
        "ad_platform",
        "ad_type",
        "target_gender",
        "target_age_group",
        "target_interests",
    ],
    "ad_events": [
        "event_id",
        "ad_id",
        "user_id",
        "timestamp",
        "day_of_week",
        "time_of_day",
        "event_type",
    ],
}


def column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    rows = cur.fetchall()
    return [r[1] for r in rows]


def validate_schema(db_path: Path) -> list[str]:
    """Return human-readable errors; empty list means OK (same rules as main())."""
    errors: list[str] = []
    if not db_path.is_file():
        return [f"database file not found: {db_path}"]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    for table, expect_cols in EXPECTED.items():
        if table not in tables:
            errors.append(f"missing table: {table}")
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            actual = column_names(conn, table)
        finally:
            conn.close()
        if actual != expect_cols:
            errors.append(
                f"column mismatch for {table}: expected {expect_cols}, got {actual}"
            )
    return errors


def main() -> int:
    root = Path(__file__).resolve().parent
    db_path = Path(os.environ.get("STREAMLIT_AD_DB_PATH", str(root / "ad_campaign_db.sqlite")))
    if not db_path.is_file():
        print(f"error: database file not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()

    ok = True
    for table, expect_cols in EXPECTED.items():
        if table not in tables:
            print(f"error: missing table: {table}", file=sys.stderr)
            ok = False
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            actual = column_names(conn, table)
        finally:
            conn.close()
        if actual != expect_cols:
            ok = False
            print(f"error: column mismatch for table {table}", file=sys.stderr)
            print(f"  expected: {expect_cols}", file=sys.stderr)
            print(f"  actual:   {actual}", file=sys.stderr)

    extra = tables - set(EXPECTED)
    if extra:
        print(f"note: extra tables (ignored): {sorted(extra)}", file=sys.stderr)

    if ok:
        print(f"ok: schema matches CSV headers for {db_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
