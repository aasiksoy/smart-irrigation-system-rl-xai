

def load_seasons(site, years, start="05-01", length=125) -> list[list[tuple[float, float]]]:
    """One season per year: `length` days of (rain, et0), starting on `start` (MM-DD)."""