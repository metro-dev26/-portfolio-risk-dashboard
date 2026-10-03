import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
from prune_releases import tags_to_prune  # noqa: E402


def test_only_dated_data_releases_older_than_the_window_are_pruned():
    tags = ["data-2026-09-01", "data-2026-09-20", "data-latest", "v1.0", "data-garbage"]
    assert tags_to_prune(tags, dt.date(2026, 10, 3), keep_days=14) == ["data-2026-09-01"]


def test_a_release_exactly_at_the_window_edge_is_kept():
    today = dt.date(2026, 10, 3)
    assert tags_to_prune(["data-2026-09-19", "data-2026-09-18"], today, keep_days=14) == ["data-2026-09-18"]


def test_a_tag_that_is_not_a_calendar_date_is_skipped_not_fatal():
    tags = ["data-2026-13-45", "data-2026-02-30", "data-2026-09-01"]
    assert tags_to_prune(tags, dt.date(2026, 10, 3)) == ["data-2026-09-01"]
