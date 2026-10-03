import datetime as dt
import json
import os
import shutil

import pytest

from risk_engine import data

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "snapshot")


def _release_fetcher(src_dir, calls):
    """Serve bundle files from a directory as if they were release assets."""
    def fetch(url):
        calls.append(url)
        with open(os.path.join(src_dir, url.rsplit("/", 1)[1]), "rb") as f:
            return f.read()
    return fetch


def _broken_fetch(url):
    raise OSError("network down")


def test_pinned_dir_wins(monkeypatch):
    snap = data.load_snapshot(use_memo=False)
    assert snap.source == "pinned"
    assert snap.as_of == "2026-06-18"
    assert "SPY" in snap.prices.columns and "AAPL" in snap.universe


def test_release_download_then_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    calls = []
    fetch = _release_fetcher(FIXTURE, calls)
    first = data.load_snapshot(release_url="https://x/rel", cache_dir=str(tmp_path),
                               fetch=fetch, use_memo=False)
    assert first.source == "release"
    assert len(calls) == 4                       # report + the three data files
    calls.clear()
    second = data.load_snapshot(release_url="https://x/rel", cache_dir=str(tmp_path),
                                fetch=fetch, use_memo=False)
    assert second.source == "cache"
    assert calls == ["https://x/rel/refresh_report.json"]   # same as_of: nothing re-downloaded


def test_new_release_replaces_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    newer = tmp_path / "newer"
    shutil.copytree(FIXTURE, newer)
    report = json.loads((newer / "refresh_report.json").read_text())
    report["as_of"] = "2026-06-19"
    (newer / "refresh_report.json").write_text(json.dumps(report))
    cache = tmp_path / "cache"
    data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                       fetch=_release_fetcher(FIXTURE, []), use_memo=False)
    snap = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_release_fetcher(str(newer), []), use_memo=False)
    assert snap.source == "release" and snap.as_of == "2026-06-19"


def test_download_failure_falls_back_with_a_visible_warning(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    snap = data.load_snapshot(cache_dir=str(tmp_path), fallback_dir=FIXTURE,
                              fetch=_broken_fetch, use_memo=False)
    assert snap.source == "fallback"
    assert "network down" in snap.warning


def test_missing_fallback_raises_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    with pytest.raises(data.SnapshotUnavailable, match="missing"):
        data.load_snapshot(cache_dir=str(tmp_path), fallback_dir=str(tmp_path / "nope"),
                           fetch=_broken_fetch, use_memo=False)


def test_staleness_uses_business_days():
    snap = data.load_snapshot(use_memo=False)       # as_of Thu 2026-06-18
    assert not data.is_stale(snap, today=dt.date(2026, 6, 22))   # Mon: 2 business days
    assert data.is_stale(snap, today=dt.date(2026, 6, 24))       # Wed: 4 business days


def test_freshness_line_names_quarantined_tickers():
    snap = data.load_snapshot(use_memo=False)
    assert data.freshness_line(snap) == "Prices as of 2026-06-18 close · 25/25 tickers"
    report = dict(snap.report, quarantined={"XOM": "suspected bad print on 2026-06-17"})
    bad = data.Snapshot(snap.prices.drop(columns=["XOM"]), snap.universe, snap.factors,
                        report, "pinned")
    assert data.freshness_line(bad) == ("Prices as of 2026-06-18 close · 24/25 tickers"
                                        " · 1 held back: XOM")
