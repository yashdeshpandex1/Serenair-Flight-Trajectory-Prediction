import pandas as pd
import requests

def get_weather_data(lat, lon, start_date, end_date):
    print(f"Fetching weather and thermodynamic data for given parameters..")
    
    def try_fetch(url, extra_params={}):
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': end_date,
            'hourly': [
                "windspeed_250hPa", "winddirection_250hPa", 
                "temperature_250hPa", "windspeed_10m", 
                "winddirection_10m", "temperature_2m"
            ],
            'wind_speed_unit': 'ms',
            'timezone': 'UTC'
        }
        
        params.update(extra_params)
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'hourly' not in data:
            return pd.DataFrame
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'wind_speed_high': data['hourly'].get('windspeed_250hPa'),
            'wind_dir_high': data['hourly'].get('winddirection_250hPa'),
            'temp_high': data['hourly'].get('temperature_250hPa'),
            'wind_speed_low': data['hourly'].get('windspeed_10m'),
            'wind_dir_low': data['hourly'].get('winddirection_10m'),
            'temp_low': data['hourly'].get('temperature_2m')
        })
        
        df['wind_speed'] = df['wind_speed_high'].fillna(df['wind_speed_low'])
        df['wind_dir'] = df['wind_dir_high'].fillna(df['wind_dir_low'])
        df['temperature'] = df['temp_high'].fillna(df['temp_low'])
        
        df = df[['timestamp', 'wind_speed', 'wind_dir', 'temperature']].dropna()
        return df
    
    df_weather = try_fetch("https://archive-api.open-meteo.com/v1/archive")
    if df_weather.empty:
        df_weather = try_fetch("https://api.open-meteo.com/v1/forecast")
        
    if df_weather.empty:
        raise ValueError('Both endpoints returned completely empty dataframes')
    
    print(f"Successfully retrieved {len(df_weather)} hours of raw thermodynamic weather data!")
    return df_weather