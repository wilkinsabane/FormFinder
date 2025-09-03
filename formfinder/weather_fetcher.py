"""Weather data fetcher using Open-Meteo API."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
from sqlalchemy.orm import Session

from .database import WeatherData, WeatherForecast, Fixture
from .config import get_config


class WeatherFetcher:
    """Fetches weather data from Open-Meteo API for match locations."""
    
    def __init__(self, session: Session):
        """Initialize WeatherFetcher with database session.
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Open-Meteo API endpoints
        self.historical_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        
        # Weather variables to fetch
        self.weather_vars = [
            "temperature_2m",
            "relative_humidity_2m", 
            "dew_point_2m",
            "apparent_temperature",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation",
            "rain",
            "snowfall",
            "cloud_cover",
            "visibility",
            "weather_code",
            "is_day",
            "shortwave_radiation"
        ]
        
        # Forecast-specific variables
        self.forecast_vars = self.weather_vars + ["precipitation_probability"]
    
    async def fetch_weather_for_fixture(self, fixture: Fixture) -> bool:
        """Fetch weather data for a specific fixture.
        
        Args:
            fixture: Fixture object with stadium location and match time
            
        Returns:
            bool: True if weather data was successfully fetched and saved
        """
        if not fixture.stadium_city or not fixture.match_date:
            self.logger.warning(f"Missing stadium city or match date for fixture {fixture.id}")
            return False
        
        # Get coordinates for stadium location
        lat, lon = await self._get_coordinates(fixture.stadium_city)
        if not lat or not lon:
            self.logger.warning(f"Could not get coordinates for {fixture.stadium_city}")
            return False
        
        # Determine if we need historical or forecast data
        now = datetime.utcnow()
        match_time = fixture.match_date
        
        if match_time < now:
            # Historical data
            return await self._fetch_historical_weather(fixture, lat, lon, match_time)
        else:
            # Forecast data
            return await self._fetch_forecast_weather(fixture, lat, lon, match_time)
    
    async def _get_coordinates(self, city_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Get latitude and longitude for a city using Open-Meteo geocoding.
        
        Args:
            city_name: Name of the city
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not found
        """
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "name": city_name,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }
                
                async with session.get(geocoding_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("results"):
                            result = data["results"][0]
                            return result.get("latitude"), result.get("longitude")
                    
                    self.logger.warning(f"Geocoding failed for {city_name}: {response.status}")
                    return None, None
                    
        except Exception as e:
            self.logger.error(f"Error geocoding {city_name}: {e}")
            return None, None
    
    async def _fetch_historical_weather(self, fixture: Fixture, lat: float, lon: float, 
                                      match_time: datetime) -> bool:
        """Fetch historical weather data for a completed match.
        
        Args:
            fixture: Fixture object
            lat: Latitude of stadium
            lon: Longitude of stadium
            match_time: Match datetime
            
        Returns:
            bool: True if data was successfully fetched and saved
        """
        try:
            # Check if we already have this data
            existing = self.session.query(WeatherData).filter_by(
                fixture_id=fixture.id,
                data_type='historical'
            ).first()
            
            if existing:
                self.logger.debug(f"Historical weather data already exists for fixture {fixture.id}")
                return True
            
            async with aiohttp.ClientSession() as session:
                # Format date for API (YYYY-MM-DD)
                date_str = match_time.strftime("%Y-%m-%d")
                
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date_str,
                    "end_date": date_str,
                    "hourly": ",".join(self.weather_vars),
                    "timezone": "UTC"
                }
                
                start_time = datetime.utcnow()
                async with session.get(self.historical_url, params=params) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._save_historical_weather_data(
                            fixture, lat, lon, match_time, data, response_time
                        )
                    else:
                        self.logger.error(f"Historical weather API error: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error fetching historical weather for fixture {fixture.id}: {e}")
            return False
    
    async def _fetch_forecast_weather(self, fixture: Fixture, lat: float, lon: float,
                                    match_time: datetime) -> bool:
        """Fetch forecast weather data for an upcoming match.
        
        Args:
            fixture: Fixture object
            lat: Latitude of stadium
            lon: Longitude of stadium
            match_time: Match datetime
            
        Returns:
            bool: True if data was successfully fetched and saved
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Calculate forecast horizon
                now = datetime.utcnow()
                horizon_hours = int((match_time - now).total_seconds() / 3600)
                
                if horizon_hours > 16 * 24:  # Open-Meteo forecast limit is 16 days
                    self.logger.warning(f"Match too far in future for forecast: {horizon_hours} hours")
                    return False
                
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": ",".join(self.forecast_vars),
                    "timezone": "UTC",
                    "forecast_days": min(16, max(1, horizon_hours // 24 + 1))
                }
                
                start_time = datetime.utcnow()
                async with session.get(self.forecast_url, params=params) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._save_forecast_weather_data(
                            fixture, lat, lon, match_time, data, response_time, horizon_hours
                        )
                    else:
                        self.logger.error(f"Forecast weather API error: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error fetching forecast weather for fixture {fixture.id}: {e}")
            return False
    
    def _save_historical_weather_data(self, fixture: Fixture, lat: float, lon: float,
                                    match_time: datetime, api_data: Dict[str, Any],
                                    response_time: float) -> bool:
        """Save historical weather data to database.
        
        Args:
            fixture: Fixture object
            lat: Latitude
            lon: Longitude
            match_time: Match datetime
            api_data: API response data
            response_time: API response time in milliseconds
            
        Returns:
            bool: True if data was saved successfully
        """
        try:
            hourly_data = api_data.get("hourly", {})
            times = hourly_data.get("time", [])
            
            if not times:
                self.logger.warning("No hourly data in API response")
                return False
            
            # Find the closest hour to match time
            match_hour = match_time.replace(minute=0, second=0, microsecond=0)
            closest_index = self._find_closest_time_index(times, match_hour)
            
            if closest_index is None:
                self.logger.warning(f"Could not find weather data for match time {match_time}")
                return False
            
            # Extract weather data for the match hour
            weather_data = WeatherData(
                fixture_id=fixture.id,
                latitude=lat,
                longitude=lon,
                weather_datetime=match_hour,
                data_type='historical',
                api_response_time_ms=response_time
            )
            
            # Set weather variables
            for var in self.weather_vars:
                if var in hourly_data and len(hourly_data[var]) > closest_index:
                    setattr(weather_data, var, hourly_data[var][closest_index])
            
            self.session.add(weather_data)
            self.session.commit()
            
            self.logger.info(f"Saved historical weather data for fixture {fixture.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving historical weather data: {e}")
            self.session.rollback()
            return False
    
    def _save_forecast_weather_data(self, fixture: Fixture, lat: float, lon: float,
                                  match_time: datetime, api_data: Dict[str, Any],
                                  response_time: float, horizon_hours: int) -> bool:
        """Save forecast weather data to database.
        
        Args:
            fixture: Fixture object
            lat: Latitude
            lon: Longitude
            match_time: Match datetime
            api_data: API response data
            response_time: API response time in milliseconds
            horizon_hours: Hours ahead of match
            
        Returns:
            bool: True if data was saved successfully
        """
        try:
            hourly_data = api_data.get("hourly", {})
            times = hourly_data.get("time", [])
            
            if not times:
                self.logger.warning("No hourly data in forecast response")
                return False
            
            # Find match time and surrounding hours for aggregation
            match_hour = match_time.replace(minute=0, second=0, microsecond=0)
            match_indices = self._find_match_period_indices(times, match_hour)
            
            if not match_indices:
                self.logger.warning(f"Could not find forecast data for match time {match_time}")
                return False
            
            # Calculate aggregated weather conditions
            forecast_data = self._calculate_forecast_aggregates(hourly_data, match_indices)
            
            # Create forecast record
            weather_forecast = WeatherForecast(
                fixture_id=fixture.id,
                latitude=lat,
                longitude=lon,
                forecast_datetime=match_hour,
                forecast_generated_at=datetime.utcnow(),
                forecast_horizon_hours=horizon_hours,
                **forecast_data
            )
            
            # Also save detailed hourly data
            for i in match_indices:
                weather_data = WeatherData(
                    fixture_id=fixture.id,
                    latitude=lat,
                    longitude=lon,
                    weather_datetime=datetime.fromisoformat(times[i].replace('Z', '+00:00')),
                    data_type='forecast',
                    api_response_time_ms=response_time
                )
                
                # Set weather variables
                for var in self.forecast_vars:
                    if var in hourly_data and len(hourly_data[var]) > i:
                        setattr(weather_data, var, hourly_data[var][i])
                
                self.session.add(weather_data)
            
            self.session.add(weather_forecast)
            self.session.commit()
            
            self.logger.info(f"Saved forecast weather data for fixture {fixture.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving forecast weather data: {e}")
            self.session.rollback()
            return False
    
    def _find_closest_time_index(self, times: List[str], target_time: datetime) -> Optional[int]:
        """Find the index of the closest time to target.
        
        Args:
            times: List of ISO time strings
            target_time: Target datetime
            
        Returns:
            Index of closest time or None if not found
        """
        try:
            target_str = target_time.isoformat()
            
            for i, time_str in enumerate(times):
                if time_str.startswith(target_str[:13]):  # Match to hour
                    return i
            
            return None
            
        except Exception:
            return None
    
    def _find_match_period_indices(self, times: List[str], match_time: datetime) -> List[int]:
        """Find indices for match period (2 hours around match time).
        
        Args:
            times: List of ISO time strings
            match_time: Match datetime
            
        Returns:
            List of indices covering match period
        """
        indices = []
        
        try:
            # Look for 2 hours around match time
            for hour_offset in [-1, 0, 1]:
                target_time = match_time + timedelta(hours=hour_offset)
                target_str = target_time.isoformat()
                
                for i, time_str in enumerate(times):
                    if time_str.startswith(target_str[:13]):  # Match to hour
                        indices.append(i)
                        break
            
            return indices
            
        except Exception:
            return []
    
    def _calculate_forecast_aggregates(self, hourly_data: Dict[str, List], 
                                     indices: List[int]) -> Dict[str, Any]:
        """Calculate aggregated weather conditions for match period.
        
        Args:
            hourly_data: Hourly weather data from API
            indices: Indices of hours to aggregate
            
        Returns:
            Dictionary of aggregated weather conditions
        """
        aggregates = {}
        
        try:
            # Temperature aggregates
            temps = [hourly_data.get("temperature_2m", [])[i] for i in indices 
                    if i < len(hourly_data.get("temperature_2m", []))]
            if temps:
                aggregates["avg_temperature"] = sum(temps) / len(temps)
                aggregates["min_temperature"] = min(temps)
                aggregates["max_temperature"] = max(temps)
            
            # Humidity
            humidity = [hourly_data.get("relative_humidity_2m", [])[i] for i in indices
                       if i < len(hourly_data.get("relative_humidity_2m", []))]
            if humidity:
                aggregates["avg_humidity"] = sum(humidity) / len(humidity)
            
            # Wind
            wind_speeds = [hourly_data.get("wind_speed_10m", [])[i] for i in indices
                          if i < len(hourly_data.get("wind_speed_10m", []))]
            if wind_speeds:
                aggregates["avg_wind_speed"] = sum(wind_speeds) / len(wind_speeds)
                aggregates["max_wind_speed"] = max(wind_speeds)
            
            # Precipitation
            precip = [hourly_data.get("precipitation", [])[i] for i in indices
                     if i < len(hourly_data.get("precipitation", []))]
            if precip:
                aggregates["total_precipitation"] = sum(precip)
            
            # Cloud cover
            clouds = [hourly_data.get("cloud_cover", [])[i] for i in indices
                     if i < len(hourly_data.get("cloud_cover", []))]
            if clouds:
                aggregates["avg_cloud_cover"] = sum(clouds) / len(clouds)
            
            # Visibility
            visibility = [hourly_data.get("visibility", [])[i] for i in indices
                         if i < len(hourly_data.get("visibility", []))]
            if visibility:
                aggregates["min_visibility"] = min(v for v in visibility if v is not None)
            
            # Weather codes (most common)
            codes = [hourly_data.get("weather_code", [])[i] for i in indices
                    if i < len(hourly_data.get("weather_code", []))]
            if codes:
                aggregates["dominant_weather_code"] = max(set(codes), key=codes.count)
            
            # Calculate derived conditions
            aggregates.update(self._calculate_condition_assessments(aggregates))
            
            return aggregates
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast aggregates: {e}")
            return {}
    
    def _calculate_condition_assessments(self, aggregates: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate playing condition assessments from weather data.
        
        Args:
            aggregates: Aggregated weather data
            
        Returns:
            Dictionary of condition assessments
        """
        assessments = {}
        
        try:
            # Precipitation risk
            total_precip = aggregates.get("total_precipitation", 0) or 0
            if total_precip == 0:
                assessments["precipitation_risk"] = "low"
            elif total_precip < 2:
                assessments["precipitation_risk"] = "medium"
            else:
                assessments["precipitation_risk"] = "high"
            
            # Wind conditions
            max_wind = aggregates.get("max_wind_speed", 0) or 0
            if max_wind < 15:
                assessments["wind_conditions"] = "calm"
            elif max_wind < 30:
                assessments["wind_conditions"] = "moderate"
            else:
                assessments["wind_conditions"] = "strong"
            
            # Overall conditions
            score = 10  # Start with perfect conditions
            
            # Deduct for precipitation
            if total_precip > 0:
                score -= min(3, total_precip)
            
            # Deduct for wind
            if max_wind > 20:
                score -= min(2, (max_wind - 20) / 10)
            
            # Deduct for extreme temperatures
            avg_temp = aggregates.get("avg_temperature", 20) or 20
            if avg_temp < 5 or avg_temp > 35:
                score -= 1
            
            # Deduct for low visibility
            min_vis = aggregates.get("min_visibility", 10000) or 10000
            if min_vis < 1000:
                score -= 2
            
            assessments["playing_conditions_score"] = max(0, score)
            
            # Overall condition categories
            if score >= 8:
                assessments["overall_conditions"] = "excellent"
                assessments["weather_impact_level"] = "minimal"
            elif score >= 6:
                assessments["overall_conditions"] = "good"
                assessments["weather_impact_level"] = "minimal"
            elif score >= 4:
                assessments["overall_conditions"] = "fair"
                assessments["weather_impact_level"] = "moderate"
            else:
                assessments["overall_conditions"] = "poor"
                assessments["weather_impact_level"] = "significant"
            
            return assessments
            
        except Exception as e:
            self.logger.error(f"Error calculating condition assessments: {e}")
            return {
                "precipitation_risk": "unknown",
                "wind_conditions": "unknown",
                "overall_conditions": "unknown",
                "weather_impact_level": "unknown",
                "playing_conditions_score": 5.0
            }
    
    async def fetch_weather_for_fixtures(self, fixtures: List[Fixture]) -> Dict[int, bool]:
        """Fetch weather data for multiple fixtures.
        
        Args:
            fixtures: List of Fixture objects
            
        Returns:
            Dictionary mapping fixture_id to success status
        """
        results = {}
        
        # Process fixtures in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(fixtures), batch_size):
            batch = fixtures[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.fetch_weather_for_fixture(fixture) for fixture in batch]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for fixture, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error fetching weather for fixture {fixture.id}: {result}")
                        results[fixture.id] = False
                    else:
                        results[fixture.id] = result
                
                # Rate limiting - wait between batches
                if i + batch_size < len(fixtures):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error processing weather batch: {e}")
                for fixture in batch:
                    results[fixture.id] = False
        
        return results