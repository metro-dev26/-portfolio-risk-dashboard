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


def test_curated_etfs_are_well_formed():
    assert config.BENCHMARK in config.CURATED_ETFS
    for t, (name, sector, asset_class) in config.CURATED_ETFS.items():
        assert t.isupper() and name and sector
        assert asset_class in ("Equity", "Bond", "Commodity")


def test_gics_map_covers_all_eleven_sectors():
    assert len(config.GICS_TO_YAHOO) == 11
    assert config.GICS_TO_YAHOO["Information Technology"] == "Technology"


def test_crises_are_ordered_windows_and_include_tariff_shock():
    labels = [c[0] for c in config.CRISES]
    assert "2025 Tariff Shock" in labels
    for _, start, end, _ in config.CRISES:
        assert start < end


def test_holding_and_window_limits():
    assert (config.MIN_HOLDINGS, config.MAX_HOLDINGS) == (2, 50)
    assert config.WINDOW_MIN_DAYS < config.WINDOW_OK_DAYS
