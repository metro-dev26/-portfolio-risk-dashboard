"""Delete dated data releases (data-YYYY-MM-DD) older than the retention window.
data-latest and any non-data release are never touched. Needs the gh CLI."""
import argparse
import datetime as dt
import json
import re
import subprocess

_DATED = re.compile(r"data-(\d{4}-\d{2}-\d{2})")


def _release_date(tag):
    """The date in a data-YYYY-MM-DD tag, or None for any other tag (including
    data-2026-13-45, which matches the shape but is not a calendar date)."""
    m = _DATED.fullmatch(tag)
    if not m:
        return None
    try:
        return dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None


def tags_to_prune(tags, today, keep_days=14):
    old = []
    for tag in tags:
        released = _release_date(tag)
        if released and (today - released).days > keep_days:
            old.append(tag)
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-days", type=int, default=14)
    args = ap.parse_args()
    listing = subprocess.run(["gh", "release", "list", "--limit", "200", "--json", "tagName"],
                             check=True, capture_output=True, text=True).stdout
    tags = [r["tagName"] for r in json.loads(listing)]
    for tag in tags_to_prune(tags, dt.date.today(), args.keep_days):
        subprocess.run(["gh", "release", "delete", tag, "-y", "--cleanup-tag"], check=True)
        print(f"deleted {tag}")


if __name__ == "__main__":
    main()
