import pandas as pd
import time
import os
from datetime import datetime
from opensky_api import OpenSkyApi
from dotenv import load_dotenv
import argparse

def get_credentials():
    """Returns credentials for OpenSky api."""
    load_dotenv()
    username = os.getenv('clientId')
    password = os.getenv('clientSecret')
    return username, password

def get_data(frequency=10, name='opensky_raw.csv',
             limit=1000000):
    
    username, password = get_credentials()
    api = OpenSkyApi(username=username, password=password)
    
    save_directory = r"../data/"
    os.makedirs(save_directory, exist_ok=True)
    filename = os.path.join(save_directory, name)
    
    total_rows_collected = 0
    pulls = 0
    
    while total_rows_collected < limit:
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current_time}] Pinging OpenSky api for flights..")
            
            states = api.get_states()
            
            if states is not None and states.states is not None:
                vector_data = []
                for s in states.states:
                    vector_data.append([
                        s.icao24, s.callsign.strip() if s.callsign else None, 
                        s.origin_country, s.time_position, s.longitude, 
                        s.latitude, s.baro_altitude, s.on_ground, s.velocity, 
                        s.true_track, s.vertical_rate, s.geo_altitude,
                        s.category, s.squawk
                    ])

                columns = ['icao24', 'callsign', 'origin_country', 
                        'timestamp', 'longitude', 'latitude', 'baro_altitude', 
                        'on_ground', 'velocity', 'true_track', 'vertical_rate', 
                        'geo_altitude', 'category', 'squawk']
                
                df = pd.DataFrame(vector_data, columns=columns)
                
                rows_in_pull = len(df)
                total_rows_collected += rows_in_pull
                pulls += 1
                
                file_exists = os.path.isfile(filename)
                df.to_csv(filename, mode='a', header=not file_exists, index=False)
                
                percentage_completed = (total_rows_collected / limit) * 100
                print(f"Success! Added {rows_in_pull} rows")
                print(f"Progress: {total_rows_collected:,} / {limit:,} ({percentage_completed:.2f}% complete)")
                
                if total_rows_collected >= limit:
                    print(f"{limit} rows collected, shutting down")
                    break   
            else:
                print("No aircrafts found.")
        except Exception as e:
            print(f"Network glitch: {e}")
            
        if total_rows_collected < limit:
            print(f"Sleeping for {frequency} seconds")
            time.sleep(frequency)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect data')
    
    parser.add_argument('--frequency', type=int, default=10, help='how often data should be collected')
    parser.add_argument('--name', type=str, default='opensky_raw.csv', help='file name')
    parser.add_argument('--limit', type=int, default=1000000, help='number of rows')
    args = parser.parse_args()
    
    get_data(
        frequency=args.frequency,
        name=args.name,
        limit=args.limit
    )