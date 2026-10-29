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
	"latitude": 52.52,
	"longitude": 13.41,
	"start_date": "2019-12-01",
	"end_date": "2019-12-07",
	"daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "et0_fao_evapotranspiration"],
    "timezone": "auto" # Time zone automatically resolved from the coordinates.
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process daily data. The order of variables needs to be the same as requested in params["daily"]
daily = response.Daily()
daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
daily_et0 = daily.Variables(4).ValuesAsNumpy()

# Generate the date range with daily time intervals
daily_data = {
	"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	),
    "temp_mean": daily.Variables(0).ValuesAsNumpy(),
    "temp_max": daily.Variables(1).ValuesAsNumpy(),
    "temp_min": daily.Variables(2).ValuesAsNumpy(),
    "precipitation": daily.Variables(3).ValuesAsNumpy(),
    "et0": daily.Variables(4).ValuesAsNumpy()
}



daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)