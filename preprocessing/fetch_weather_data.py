import pandas as pd
import requests

def get_weather_data(lat, lon, start_date, end_date):
    print(f"Fetching weather and thermodynamic data for given parameters..")
    start_str = str(start_date)[:10]
    end_str = str(end_date)[:10]
    def try_fetch(url, extra_params={}, fallback_surface_only=False):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str, 
            "end_date": end_str,
            "wind_speed_unit": "ms", 
            "timezone": "UTC"
        }
        
        if fallback_surface_only:
             params["hourly"] = ["windspeed_10m", "winddirection_10m", "temperature_2m"]
        else:
             params["hourly"] = [
                 "windspeed_250hPa", "winddirection_250hPa", "temperature_250hPa",
                 "windspeed_10m", "winddirection_10m", "temperature_2m"
             ]
             
        params.update(extra_params)
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'error' in data:
            print(f"  [API Error from {url}]: {data.get('reason')}")
            return pd.DataFrame()
            
        if 'hourly' not in data:
            print(f"  [Unknown Error]: No 'hourly' data found in response.")
            return pd.DataFrame()

        if fallback_surface_only:
             df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['hourly']['time']),
                'wind_speed': data['hourly']['windspeed_10m'],
                'wind_dir': data['hourly']['winddirection_10m'],
                'temperature': data['hourly']['temperature_2m']
            })
        else:
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
            df = df[['timestamp', 'wind_speed', 'wind_dir', 'temperature']]
            
        return df.dropna()
    
    # The Historical Archive (Best for data > 5 days old)
    print("Trying Archive API...")
    df_weather = try_fetch("https://archive-api.open-meteo.com/v1/archive")
    
    # The Live GFS Forecast Model (Best for data < 5 days old)
    if df_weather.empty:
        print("Archive failed. Trying Live GFS Forecast Model...")
        df_weather = try_fetch("https://api.open-meteo.com/v1/forecast", extra_params={"models": "gfs_seamless"})

    # (Surface weather only)
    if df_weather.empty:
        print("GFS failed. Trying basic surface weather fallback...")
        df_weather = try_fetch("https://api.open-meteo.com/v1/forecast", fallback_surface_only=True)

    if df_weather.empty:
        raise ValueError("Critical Error: All Open-Meteo endpoints failed. Check the [API Error] prints above to see why.")
        
    print(f"Successfully retrieved {len(df_weather)} hours of thermodynamic weather data!")
    return df_weather