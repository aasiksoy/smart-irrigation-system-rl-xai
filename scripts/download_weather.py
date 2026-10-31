"""Download daily Open-Meteo weather (1990-2023) for each project site.
Run from the repo root:  python -m scripts.download_weather
Output: data/weather_<site>.csv with columns date, tmax, tmin, rain, et0
"""
import os

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "1990-01-01"
END_DATE = "2023-12-31"
VARIABLES = ["temperature_2m_max", "temperature_2m_min",
             "precipitation_sum", "et0_fao_evapotranspiration"]

SITES = {
    "leuven": (50.88, 4.70), 
    "nicosia": (35.17, 33.36),
    "seville": (37.39, -5.98), 
}

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def download_site(name, lat, lon):
    """Download one site's daily weather, check it, and save it to data/weather_<name>.csv."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": VARIABLES,
        "timezone": "auto",
    }
    response = openmeteo.weather_api(URL, params=params)[0]
    daily = response.Daily()

    offset = response.UtcOffsetSeconds()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time() + offset, unit="s"),
        end=pd.to_datetime(daily.TimeEnd() + offset, unit="s"),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )

    df = pd.DataFrame({
        "date": dates.date,
        "tmax": daily.Variables(0).ValuesAsNumpy(),
        "tmin": daily.Variables(1).ValuesAsNumpy(),
        "rain": daily.Variables(2).ValuesAsNumpy(),
        "et0": daily.Variables(3).ValuesAsNumpy(),
    })


    print(f"\n{name}: {df['date'].iloc[0]} to {df['date'].iloc[-1]}, {len(df)} days")
    missing = df.isna().sum()
    if missing.sum() > 0:
        print("warning, missing values:\n", missing[missing > 0])

    july = df[pd.to_datetime(df["date"]).dt.month == 7]
    print(f"July average: tmax {july['tmax'].mean():.1f} °C, "
          f"ET0 {july['et0'].mean():.1f} mm/day, rain {july['rain'].mean():.1f} mm/day")

    path = f"data/weather_{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved to {path}")


def main():
    os.makedirs("data", exist_ok=True)
    for name, (lat, lon) in SITES.items():
        download_site(name, lat, lon)


if __name__ == "__main__":
    main()