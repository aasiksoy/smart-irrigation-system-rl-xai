import pandas as pd

def load_seasons(site, years, start="05-01", length=125) -> list[list[tuple[float, float]]]:
    """One season per year: `length` days of (rain, et0), starting on `start` (MM-DD).
    site:   which weather file to read
    years:  which years to cut out
    """
    df = pd.read_csv(f"data/weather_{site}.csv", parse_dates=["date"])

    seasons = []
    for year in years:
        after_start = df[df["date"] >= f"{year}-{start}"] 
        season_df = after_start.head(length)
        season = [(float(rain), float(et0))
                  for rain, et0 in zip(season_df["rain"], season_df["et0"])]
        seasons.append(season)

    return seasons