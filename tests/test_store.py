from api import store


def test_save_get_roundtrip(tmp_path):
    db = str(tmp_path / "t.db")
    pid = store.save_portfolio("mine", {"AAPL": 1000, "MSFT": 2000}, path=db)
    row = store.get_portfolio(pid, path=db)
    assert row["name"] == "mine"
    assert row["holdings"] == {"AAPL": 1000, "MSFT": 2000}
    assert row["id"] == pid and row["created_at"]


def test_get_missing_returns_none(tmp_path):
    db = str(tmp_path / "t.db")
    store.save_portfolio("x", {"AAPL": 1, "MSFT": 1}, path=db)
    assert store.get_portfolio(999, path=db) is None


def test_list_returns_all_in_order(tmp_path):
    db = str(tmp_path / "t.db")
    store.save_portfolio("a", {"AAPL": 1, "MSFT": 1}, path=db)
    store.save_portfolio("b", {"MSFT": 2, "JPM": 1}, path=db)
    assert [r["name"] for r in store.list_portfolios(path=db)] == ["a", "b"]
