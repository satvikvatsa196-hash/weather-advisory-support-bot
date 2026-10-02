import httpx
from .models import (
    Location,
    WeatherData,
    GeocodingError,
    UnknownLocationError,
    WeatherAPIError,
    MalformedResponseError,
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

async def get_location(city_name: str) -> Location:
    """
    Geocode a city name to get its latitude and longitude.
    """
    params = {"name": city_name, "count": 1}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(GEOCODING_URL, params=params)
            response.raise_for_status()
    except httpx.TimeoutException as e:
        raise GeocodingError("Geocoding API timeout") from e
    except httpx.HTTPError as e:
        raise GeocodingError(f"Geocoding API error: {e}") from e

    try:
        data = response.json()
    except ValueError as e:
        raise MalformedResponseError("Invalid JSON from geocoding API") from e

    results = data.get("results")
    if not results or len(results) == 0:
        raise UnknownLocationError(f"Location not found: {city_name}")

    first_result = results[0]
    try:
        return Location(
            name=first_result["name"],
            latitude=first_result["latitude"],
            longitude=first_result["longitude"],
        )
    except KeyError as e:
        raise MalformedResponseError(f"Missing expected field in geocoding response: {e}") from e


async def get_weather(lat: float, lon: float) -> WeatherData:
    """
    Fetch current weather data for a given latitude and longitude.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,precipitation,precipitation_probability,uv_index",
        "hourly": "temperature_2m,precipitation_probability,uv_index",
        "daily": "temperature_2m_max,precipitation_sum,uv_index_max",
        "timezone": "auto"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(FORECAST_URL, params=params)
            response.raise_for_status()
    except httpx.TimeoutException as e:
        raise WeatherAPIError("Weather API timeout") from e
    except httpx.HTTPError as e:
        raise WeatherAPIError(f"Weather API error: {e}") from e

    try:
        data = response.json()
    except ValueError as e:
        raise MalformedResponseError("Invalid JSON from weather API") from e

    current = data.get("current")
    if not current:
        raise MalformedResponseError("Missing 'current' object in weather response")

    try:
        # Strict mapping. Missing critical fields will raise KeyError -> MalformedResponseError
        return WeatherData(
            temperature_2m=current["temperature_2m"],
            wind_speed_10m=current["wind_speed_10m"],
            precipitation=current["precipitation"],
            precipitation_probability=current.get("precipitation_probability"),
            uv_index=current.get("uv_index"),
            hourly=data.get("hourly"),
            daily=data.get("daily"),
        )
    except KeyError as e:
        raise MalformedResponseError(f"Missing expected field in weather response: {e}") from e
