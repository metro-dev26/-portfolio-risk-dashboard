import datetime as dt
import json
import os
import shutil
import time

import numpy as np
import pytest

from risk_engine import config, data

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


def _release_dir(tmp_path, name, as_of):
    """A copy of the fixture bundle stamped with a different as_of."""
    out = tmp_path / name
    shutil.copytree(FIXTURE, out)
    report = json.loads((out / "refresh_report.json").read_text())
    report["as_of"] = as_of
    (out / "refresh_report.json").write_text(json.dumps(report))
    return str(out)


def test_pinned_dir_wins():
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


def test_download_failure_prefers_the_last_downloaded_copy(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"
    newer = _release_dir(tmp_path, "newer", "2026-06-19")
    first = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(newer, []), fallback_dir=FIXTURE,
                               use_memo=False)
    assert first.source == "release" and first.warning is None

    snap = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_broken_fetch, fallback_dir=FIXTURE, use_memo=False)
    assert snap.source == "cache" and snap.as_of == "2026-06-19"
    assert "network down" in snap.warning and "2026-06-19" in snap.warning
    assert "last downloaded copy" in snap.warning


def test_unreadable_cache_falls_through_to_the_bundled_copy(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"
    data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                       fetch=_release_fetcher(FIXTURE, []), fallback_dir=FIXTURE, use_memo=False)
    (cache / "prices.csv.gz").write_bytes(b"not-gzip")

    snap = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_broken_fetch, fallback_dir=FIXTURE, use_memo=False)
    assert snap.source == "fallback"
    assert "network down" in snap.warning and "bundled copy" in snap.warning


def test_cached_copy_is_retried_on_the_fallback_schedule(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    monkeypatch.setattr(data, "_MEMO", {})
    cache = tmp_path / "cache"
    data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                       fetch=_release_fetcher(FIXTURE, []), fallback_dir=FIXTURE, use_memo=False)

    down = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_broken_fetch, fallback_dir=FIXTURE, max_age_s=3600)
    assert down.source == "cache"

    calls = []
    soon = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_release_fetcher(FIXTURE, calls), fallback_dir=FIXTURE,
                              max_age_s=3600, clock=lambda: time.time() + 10)
    assert soon.source == "cache" and calls == []        # inside the retry window: memoized

    later = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(FIXTURE, calls), fallback_dir=FIXTURE,
                               max_age_s=3600, clock=lambda: time.time() + 301)
    assert calls and later.source == "cache" and later.warning is None   # retried, same as_of


def test_missing_fallback_raises_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    with pytest.raises(data.SnapshotUnavailable, match="missing"):
        data.load_snapshot(cache_dir=str(tmp_path), fallback_dir=str(tmp_path / "nope"),
                           fetch=_broken_fetch, use_memo=False)


def test_staleness_uses_business_days():
    snap = data.load_snapshot(use_memo=False)       # as_of Thu 2026-06-18
    assert not data.is_stale(snap, today=dt.date(2026, 6, 22))   # Mon: 2 business days
    assert not data.is_stale(snap, today=dt.date(2026, 6, 23))   # Tue: exactly 3 business days (not stale)
    assert data.is_stale(snap, today=dt.date(2026, 6, 24))       # Wed: 4 business days


def test_freshness_line_names_quarantined_tickers():
    snap = data.load_snapshot(use_memo=False)
    assert data.freshness_line(snap) == "Prices as of 2026-06-18 close · 25/25 tickers"
    report = dict(snap.report, quarantined={"XOM": "suspected bad print on 2026-06-17"})
    bad = data.Snapshot(snap.prices.drop(columns=["XOM"]), snap.universe, snap.factors,
                        report, "pinned")
    assert data.freshness_line(bad) == ("Prices as of 2026-06-18 close · 24/25 tickers"
                                        " · 1 held back: XOM")


def test_corrupt_download_does_not_poison_cache(monkeypatch, tmp_path):
    """Load 1: release A. Load 2: release B with corrupt prices → cached A. Load 3: release B good."""
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"

    # Load 1: good release A (as_of 2026-06-18)
    calls = []
    snap1 = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(FIXTURE, calls), use_memo=False)
    assert snap1.source == "release" and snap1.as_of == "2026-06-18"

    # Load 2: release B (as_of 2026-06-19) with corrupt prices.csv.gz → last good download
    corrupt_dir = tmp_path / "corrupt"
    shutil.copytree(FIXTURE, corrupt_dir)
    report = json.loads((corrupt_dir / "refresh_report.json").read_text())
    report["as_of"] = "2026-06-19"
    (corrupt_dir / "refresh_report.json").write_text(json.dumps(report))
    (corrupt_dir / "prices.csv.gz").write_bytes(b"not-gzip")
    calls.clear()
    snap2 = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(str(corrupt_dir), calls), use_memo=False)
    assert snap2.source == "cache" and snap2.as_of == "2026-06-18"

    # Load 3: release B good (as_of 2026-06-19) → should return release
    good_b_dir = tmp_path / "good_b"
    shutil.copytree(FIXTURE, good_b_dir)
    report = json.loads((good_b_dir / "refresh_report.json").read_text())
    report["as_of"] = "2026-06-19"
    (good_b_dir / "refresh_report.json").write_text(json.dumps(report))
    calls.clear()
    snap3 = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(str(good_b_dir), calls), use_memo=False)
    assert snap3.source == "release" and snap3.as_of == "2026-06-19"


def test_corrupt_cached_refresh_report_forces_redownload(monkeypatch, tmp_path):
    """Cache report corrupted → next load re-downloads."""
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"

    # Load 1: good release
    data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                       fetch=_release_fetcher(FIXTURE, []), use_memo=False)

    # Corrupt the cached refresh_report.json
    (tmp_path / "cache" / "refresh_report.json").write_bytes(b"{not json")

    # Load 2: should detect corruption and re-download
    calls = []
    snap = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_release_fetcher(FIXTURE, calls), use_memo=False)
    assert snap.source == "release"
    assert len(calls) == 4  # all four files re-downloaded


def test_memo_keyed_by_arguments_and_respects_fallback_timeout(monkeypatch, tmp_path):
    """Memo expires faster for fallback than for release/cache."""
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"

    # Reset the memo to start clean
    monkeypatch.setattr(data, "_MEMO", {})

    # Load 1: network fails → fallback, memo'd for 300s
    calls = []
    snap1 = data.load_snapshot(cache_dir=str(cache), fallback_dir=FIXTURE,
                               fetch=_broken_fetch, use_memo=True, max_age_s=3600)
    assert snap1.source == "fallback"

    # Load 2: 10s later with working fetch (via clock) → still returns memoized fallback
    snap2 = data.load_snapshot(cache_dir=str(cache), fallback_dir=FIXTURE,
                               fetch=_release_fetcher(FIXTURE, calls), use_memo=True,
                               max_age_s=3600, clock=lambda: time.time() + 10)
    assert snap2.source == "fallback"
    assert calls == []  # never tried to fetch

    # Load 3: 301s later → memo expired, returns "release"
    calls.clear()
    snap3 = data.load_snapshot(cache_dir=str(cache), fallback_dir=FIXTURE,
                               fetch=_release_fetcher(FIXTURE, calls), use_memo=True,
                               max_age_s=3600, clock=lambda: time.time() + 301)
    assert snap3.source == "release"


def test_corrupt_file_parsing_raises_unavailable(monkeypatch, tmp_path):
    """Corrupt universe.json in fallback → SnapshotUnavailable naming both failures."""
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")

    # Create a fallback dir with corrupt universe.json
    bad_fallback = tmp_path / "bad_fallback"
    shutil.copytree(FIXTURE, bad_fallback)
    (bad_fallback / "universe.json").write_text("{not json")

    # Try to load with broken fetch (so release fails) and bad fallback
    with pytest.raises(data.SnapshotUnavailable) as exc_info:
        data.load_snapshot(cache_dir=str(tmp_path), fallback_dir=str(bad_fallback),
                           fetch=_broken_fetch, use_memo=False)
    msg = str(exc_info.value)
    assert "network down" in msg  # release failure reason
    assert "unreadable" in msg or "universe.json" in msg  # fallback failure reason


def test_corrupt_gzip_deflate_body_not_poison_cache(monkeypatch, tmp_path):
    """Cached prices.csv.gz with valid header but corrupt deflate body → re-download, not fallback."""
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"

    # Load 1: good release
    snap1 = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(FIXTURE, []), use_memo=False)
    assert snap1.source == "release"

    # Corrupt the cached prices.csv.gz: keep header but corrupt deflate body
    cached_prices = cache / "prices.csv.gz"
    original = cached_prices.read_bytes()
    # Corrupt bytes 20-30 (middle of deflate stream)
    corrupted = original[:20] + b"\x00\xFF" * 5 + original[30:]
    cached_prices.write_bytes(corrupted)

    # Load 2: fetch is working, but cache is corrupt → must re-download (not fallback)
    calls = []
    snap2 = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                               fetch=_release_fetcher(FIXTURE, calls), use_memo=False)
    assert snap2.source == "release"
    assert len(calls) == 4  # All 4 files re-downloaded


def test_read_bundle_on_corrupt_gzip_deflate(tmp_path):
    """read_bundle on corrupt deflate body → SnapshotUnavailable, not zlib.error."""
    import gzip
    bad_dir = tmp_path / "bad"
    shutil.copytree(FIXTURE, bad_dir)

    # Corrupt deflate body while keeping gzip header
    cached_prices = bad_dir / "prices.csv.gz"
    original = cached_prices.read_bytes()
    corrupted = original[:20] + b"\x00\xFF" * 5 + original[30:]
    cached_prices.write_bytes(corrupted)

    # read_bundle must raise SnapshotUnavailable, not zlib.error
    with pytest.raises(data.SnapshotUnavailable, match="unreadable"):
        data.read_bundle(str(bad_dir), "test")


def test_read_bundle_on_non_dict_report(tmp_path):
    """read_bundle with refresh_report.json = 5 → SnapshotUnavailable, not TypeError."""
    bad_dir = tmp_path / "bad"
    shutil.copytree(FIXTURE, bad_dir)
    (bad_dir / "refresh_report.json").write_text("5")

    # read_bundle must raise SnapshotUnavailable, not TypeError
    with pytest.raises(data.SnapshotUnavailable, match="unreadable|dict"):
        data.read_bundle(str(bad_dir), "test")


def test_unreadable_cache_and_fallback_name_every_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    cache = tmp_path / "cache"
    data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                       fetch=_release_fetcher(FIXTURE, []), fallback_dir=FIXTURE, use_memo=False)
    (cache / "prices.csv.gz").write_bytes(b"not-gzip")
    with pytest.raises(data.SnapshotUnavailable) as exc_info:
        data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache), fetch=_broken_fetch,
                           fallback_dir=str(tmp_path / "nope"), use_memo=False)
    msg = str(exc_info.value)
    assert "network down" in msg                       # the download
    assert str(cache) in msg and "prices.csv.gz" in msg  # the cached copy
    assert "nope" in msg                               # the bundled copy


def test_bundled_fallback_reads_cleanly():
    """data/fallback/ is replaced by newer bundles over time, so nothing here pins a
    ticker count or a date."""
    snap = data.read_bundle(data.FALLBACK_DIR, "fallback")
    assert config.BENCHMARK in snap.prices.columns
    dt.date.fromisoformat(snap.as_of)
    last = snap.prices.ffill().iloc[-1]
    assert np.isfinite(last).all() and (last > 0).all()
