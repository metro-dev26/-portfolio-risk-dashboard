from risk_engine import config


def test_tickers_are_unique_and_nonempty():
    assert len(config.TICKERS) == 24
    assert len(set(config.TICKERS)) == len(config.TICKERS)


def test_bonds_are_subset_of_tickers():
    assert config.BONDS.issubset(set(config.TICKERS))


def test_every_ticker_has_a_sector_and_asset_class():
    for t in config.TICKERS:
        assert t in config.SECTOR
        assert config.ASSET_CLASS[t] in ("Bond", "Equity")


def test_trading_days_constant():
    assert config.TRADING_DAYS == 252
