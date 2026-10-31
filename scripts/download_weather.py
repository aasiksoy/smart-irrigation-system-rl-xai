import os
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 50.88,
	"longitude": 4.70, 
    "start_date": "1990-01-01",
    "end_date": "2023-12-31",  
	"daily": ["temperature_2m_max",
			 "temperature_2m_min",
			 "precipitation_sum",
			 "et0_fao_evapotranspiration"],
    "timezone": "auto" # Time zone automatically resolved from the coordinates.
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")

# Process daily data. The order of variables needs to be the same as requested in params["daily"]
daily = response.Daily()

offset = response.UtcOffsetSeconds()

dates = pd.date_range(
    start=pd.to_datetime(daily.Time() + offset, unit="s"),
    end=pd.to_datetime(daily.TimeEnd() + offset, unit="s"),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left",
)
dates = pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left",
)

df = pd.DataFrame({
    "date": dates.date,
    "tmax": daily.Variables(0).ValuesAsNumpy(),
    "tmin": daily.Variables(1).ValuesAsNumpy(), 
    "rain": daily.Variables(2).ValuesAsNumpy(),
    "et0":  daily.Variables(3).ValuesAsNumpy(),
})


print(df.head())
print(df.tail())
print("Missing values:\n", df.isna().sum())

os.makedirs("data", exist_ok=True) 
df.to_csv("data/weather_leuven.csv", index=False) 
print(f"Saved {len(df)} days to data/weather_leuven.csv")