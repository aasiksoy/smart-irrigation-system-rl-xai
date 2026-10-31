"""Tests for simulator/weather.py.  Run: pytest tests/test_weather.py -v"""

from simulator.weather import load_seasons

SITES = ["leuven", "nicosia", "seville"]


def test_every_season_has_125_days():
    for site in SITES:
        seasons = load_seasons(site, range(1995, 2000))
        for season in seasons:
            assert len(season) == 125, f"{site}: season has {len(season)} days, expected 125"


def test_one_season_per_year():
    seasons = load_seasons("leuven", range(1990, 2000))
    assert len(seasons) == 10, f"expected 10 seasons, got {len(seasons)}"


def test_no_negative_or_missing_values():
    for site in SITES:
        for season in load_seasons(site, range(1995, 2000)):
            for rain, et0 in season:
                assert rain >= 0, f"{site}: invalid rain value {rain}"
                assert et0 >= 0, f"{site}: invalid ET0 value {et0}"


def test_et0_is_plausible():
    for site in SITES:
        for season in load_seasons(site, range(1995, 2000)):
            for _, et0 in season:
                assert 0 <= et0 <= 15, f"{site}: implausible ET0 {et0} mm/day"


def test_length_is_respected():
    seasons = load_seasons("leuven", [2000], length=30)
    assert len(seasons) == 1, "expected exactly one season"
    assert len(seasons[0]) == 30, f"expected 30 days, got {len(seasons[0])}"