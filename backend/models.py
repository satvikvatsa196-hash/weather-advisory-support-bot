from pydantic import BaseModel
from typing import Optional

class Location(BaseModel):
    name: str
    latitude: float
    longitude: float

class WeatherData(BaseModel):
    temperature_2m: float
    wind_speed_10m: float
    precipitation: float
    precipitation_probability: Optional[float] = None
    uv_index: Optional[float] = None
    hourly: Optional[dict] = None
    daily: Optional[dict] = None

class GeocodingError(Exception):
    """Raised when the geocoding API fails or returns invalid data."""
    pass

class UnknownLocationError(Exception):
    """Raised when a city cannot be found."""
    pass

class WeatherAPIError(Exception):
    """Raised when the weather forecast API fails."""
    pass

class MalformedResponseError(Exception):
    """Raised when the API response is missing expected fields."""
    pass
