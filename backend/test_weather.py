import pytest
from unittest.mock import patch, AsyncMock
import httpx
from backend.weather import get_location, get_weather
from backend.models import (
    Location,
    WeatherData,
    GeocodingError,
    UnknownLocationError,
    WeatherAPIError,
    MalformedResponseError,
)

# --- Mocks ---

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        if self._json_data is None:
            raise ValueError("Invalid JSON")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("Error", request=None, response=self)

# --- Geocoding Tests ---

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_location_success(mock_get):
    mock_get.return_value = MockResponse({
        "results": [
            {"name": "Bhopal", "latitude": 23.25, "longitude": 77.41667}
        ]
    })
    loc = await get_location("Bhopal")
    assert loc.name == "Bhopal"
    assert loc.latitude == 23.25
    assert loc.longitude == 77.41667

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_location_unknown(mock_get):
    mock_get.return_value = MockResponse({"results": []})
    with pytest.raises(UnknownLocationError, match="Location not found: FakeCity"):
        await get_location("FakeCity")

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_location_http_error(mock_get):
    mock_get.side_effect = httpx.HTTPError("Network failure")
    with pytest.raises(GeocodingError, match="Geocoding API error"):
        await get_location("Bhopal")

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_location_malformed(mock_get):
    # Missing 'latitude' in result
    mock_get.return_value = MockResponse({
        "results": [
            {"name": "Bhopal", "longitude": 77.41667}
        ]
    })
    with pytest.raises(MalformedResponseError, match="Missing expected field"):
        await get_location("Bhopal")

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_location_invalid_json(mock_get):
    mock_get.return_value = MockResponse(None)
    with pytest.raises(MalformedResponseError, match="Invalid JSON"):
        await get_location("Bhopal")

# --- Weather Tests ---

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_weather_success(mock_get):
    mock_get.return_value = MockResponse({
        "current": {
            "temperature_2m": 30.5,
            "wind_speed_10m": 12.0,
            "precipitation": 0.0,
            "precipitation_probability": 10.0,
            "uv_index": 7.5
        },
        "hourly": {"time": [], "temperature_2m": []},
        "daily": {"time": [], "temperature_2m_max": []}
    })
    weather = await get_weather(23.25, 77.41667)
    assert weather.temperature_2m == 30.5
    assert weather.wind_speed_10m == 12.0
    assert weather.precipitation == 0.0
    assert weather.precipitation_probability == 10.0
    assert weather.uv_index == 7.5
    assert weather.hourly is not None
    assert weather.daily is not None

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_weather_http_error(mock_get):
    mock_get.side_effect = httpx.TimeoutException("Timeout")
    with pytest.raises(WeatherAPIError, match="Weather API timeout"):
        await get_weather(23.25, 77.41667)

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_weather_missing_current(mock_get):
    mock_get.return_value = MockResponse({"hourly": {}})
    with pytest.raises(MalformedResponseError, match="Missing 'current' object"):
        await get_weather(23.25, 77.41667)

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_weather_missing_fields(mock_get):
    mock_get.return_value = MockResponse({
        "current": {
            "temperature_2m": 30.5,
            # Missing wind_speed_10m and precipitation which are strictly required
        }
    })
    with pytest.raises(MalformedResponseError, match="Missing expected field"):
        await get_weather(23.25, 77.41667)

@pytest.mark.asyncio
@patch("backend.weather.httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_get_weather_optional_fields(mock_get):
    # Missing optional fields like uv_index and precipitation_probability
    mock_get.return_value = MockResponse({
        "current": {
            "temperature_2m": 30.5,
            "wind_speed_10m": 12.0,
            "precipitation": 0.0,
        },
        "hourly": {},
        "daily": {}
    })
    weather = await get_weather(23.25, 77.41667)
    assert weather.uv_index is None
    assert weather.precipitation_probability is None
