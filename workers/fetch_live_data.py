import pandas as pd
import requests
import logging
from dotenv import load_dotenv
import os, time, sys
from datetime import datetime, timedelta


# from opensky api itself, refreshes the token if it expires
class TokenManager:
    def __init__(self):
        self.token = None
        self.expires_at = None
    
    def get_token(self):
        """Return a valid access token, refreshing automatically if needed."""
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()
    
    def _refresh(self):
        load_dotenv('../.env')
        client_id = os.getenv('clientId')
        client_secret = os.getenv('clientSecret')
        
        TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
        
        if not client_id or not client_secret:
            return None
        
        r = requests.post(
            TOKEN_URL,
            data = {
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret
            }, timeout = 10
        )
        r.raise_for_status()
        data = r.json()
        self.token = data['access_token']
        self.expires_at = datetime.now() + timedelta(seconds=data.get('expires_in', 1800) - 30)
        return self.token

    def headers(self):
        """Return request headers with a valid Beared token AND the critical custom User-Agent."""
        headers = {"User-Agent": "Serenair-Live-Worker/1.0"}
        token = self.get_token()
        if token:
            headers['Authorization'] = f"Bearer {token}"
        return headers
    
tokens = TokenManager()


def fetch_live_flights_data():
    logger = logging.getLogger(__name__)
    url = "https://opensky-network.org/api/states/all"
    
    columns = ['icao24', 'callsign', 'origin_country',
               'timestamp', 'longitude', 'latitude', 'baro_altitude',
               'on_ground', 'velocity', 'true_track', 'vertical_rate',
               'geo_altitude', 'category', 'squawk']
    
    try:
        response = requests.get(url, headers=tokens.headers(), timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.exception(f"Failed to fetch states: {e}")
        return pd.DataFrame(columns=columns)
    
    if not data or 'states' not in data or data['states'] is None:
        return pd.DataFrame(columns=columns)
    
    vector_data = [
        [
            s[0],                                 # icao24
            str(s[1]).strip() if s[1] else None,  # callsign  
            s[2],                                 # origin_country
            s[3],                                 # time_position / timestamp
            s[5],                                 # longitude
            s[6],                                 # latitude
            s[7],                                 # baro_altitude
            s[8],                                 # on_ground
            s[9],                                 # velocity
            s[10],                                # true_track
            s[11],                                # vertical_rate
            s[13],                                # geo_altitude  
            s[17] if len(s) > 17 else None,       # category
            s[14]                                 # squawk
        ] for s in data['states']
    ]
    
    df = pd.DataFrame(vector_data, columns=columns)
    df = df.dropna(subset=['timestamp', 'latitude', 'longitude'])
    df = df[df['on_ground'] == False].copy()
    df['timestamp'] = df['timestamp'].astype(int)
    
    return df


def fetch_current_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    
    def try_fetch(extra_params={}, fallback_surface_only=False):
        params = {
            'latitude': lat,
            'longitude': lon,
            'wind_speed_unit': 'ms',
            'timezone': 'UTC'
        }
        
        if fallback_surface_only:
            params['current'] = [
                'temperature_2m', 'wind_speed_10m', 'wind_direction_10m'
            ]
        else:
            params['current'] = [
                'temperature_250hPa', 'windspeed_250hPa', 'winddirection_250hPa',
                'temperature_2m', 'wind_speed_10m', 'wind_direction_10m'
            ]
        
        params.update(extra_params)
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f" [Network Error]: {e}")
            return None
        
        if 'error' in data:
            print(f" [API Error]: {data.get('reason')}")
            return None
        
        if 'current' not in data:
            return None
        
        current_data = data['current']
        
        if fallback_surface_only:
            return (
                current_data.get('temperature_2m'),
                current_data.get('wind_speed_10m'),
                current_data.get('wind_direction_10m')
            )
        else:
            temp = current_data.get('temperature_250hPa')
            if temp is None:
                temp = current_data.get('temperature_2m')
                
            wind_speed = current_data.get('windspeed_250hPa')
            if wind_speed is None:
                wind_speed = current_data.get('wind_speed_10m')
                
            wind_dir = current_data.get('winddirection_250hPa')
            if wind_dir is None:
                wind_dir = current_data.get('wind_direction_10m')
            
            return temp, wind_speed, wind_dir
    result = try_fetch(extra_params={'models': 'gfs_seamless'})
    
    if result is None or None in result:
        print(f"GFS high_altitude failed. Trying basic surface weather fallback..")
        result = try_fetch(fallback_surface_only=True)
        
    if result is None or None in result:
        print("Critical Error: All Open-Meteo current endpoints failed.")
        return None, None, None
        
    return result


if __name__ == "__main__":
    flights_df = fetch_live_flights_data()
    
    if not flights_df.empty:
        print(f"Retrieved {len(flights_df)} active flights.")
        print("\n--- Sample Flight ---")
        sample_flight = flights_df.iloc[0]
        print(sample_flight[['icao24', 'callsign', 'latitude', 'longitude', 'velocity']])
        
        print("\n Testing Open-Meteo Weather Fetcher for this coordinate..")
        lat = sample_flight['latitude']
        lon = sample_flight['longitude']
        
        temp, wind_speed, wind_dir = fetch_current_weather(lat, lon)
        if temp is not None:
            print(f"   Weather at ({lat:.2f}, {lon:.2f}):")
            print(f"   Temperature: {temp}°C")
            print(f"   Wind Speed : {wind_speed} m/s")
            print(f"   Wind Dir   : {wind_dir}°")
        else:
            print("Weather fetching failed.")
    else:
        print("Failed to retrieve flight data or no flights in the air.")