import numpy as np
import psycopg
from datetime import datetime
import time
from database_conn import get_connection_uri
import pandas as pd
from fetch_live_data import fetch_current_weather


def prod_database_ingestion(df, conn_string):
    if df.empty:
        return 
    
    unique_aircraft = df[['icao24', 'origin_country', 'category']].drop_duplicates()
    unique_aircraft = unique_aircraft.replace({np.nan: None})
    
    states_columns = [
        'icao24', 'callsign', 'timestamp', 'latitude', 
        'longitude', 'baro_altitude', 'velocity', 
        'vertical_rate', 'true_track', 'geo_altitude',
        'on_ground', 'squawk'
    ]
    df_states = df[states_columns].replace({np.nan: None})
    
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                insert_aircraft_query = """
                    INSERT INTO aircraft (icao24, origin_country, category)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (icao24) DO NOTHING;
                """
                cur.executemany(insert_aircraft_query, unique_aircraft.values.tolist())
                
                copy_query = f"""
                    COPY aircraft_states ({', '.join(states_columns)})
                    FROM STDIN;
                """
                with cur.copy(copy_query) as copy:
                    for row in df_states.itertuples(index=False, name=None):
                        copy.write_row(row)
                        
                conn.commit()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(df_states)} vectors_ingested.")
    except Exception as e:
        print(f"Database error: {e}")
        
        
def prune_old_data(conn_string, tolerance_mins=15):
    try:
        cutoff = int(time.time()) - (tolerance_mins * 60)
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM aircraft_states WHERE timestamp < %s", (cutoff,))
                deleted_count = cur.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    print(f"Pruned {deleted_count} old GPS pings from the database.")
    except Exception as e:
        print(f"Pruning failed: {e}")
        
      
def fetch_and_integrate_data(continent='europe'):
    bbox = get_bbox(continent_name=continent)
    min_lat, max_lat, min_lon, max_lon = bbox
    conn_string = get_connection_uri()
    
    query = """
        SELECT 
            s.icao24, s.callsign, s.timestamp, s.longitude,
            s.latitude, s.baro_altitude, s.on_ground, s.velocity,
            s.true_track, s.vertical_rate, s.geo_altitude,
            a.category, s.squawk
        FROM aircraft_states s
        JOIN aircraft a on s.icao24 = a.icao24
            WHERE s.latitude between %s and %s
            AND s.longitude between %s and %s
        ORDER BY s.timestamp ASC;
    """
    
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (min_lat, max_lat, min_lon, max_lon))
                rows = cur.fetchall()
                
                if not rows:
                    print(f"No flights records found in database for {continent}.")
                    return pd.DataFrame()
                
                colnames = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        print(f"Pulled {len(df)} flight records from the database.")
        
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        
        temp, wind_speed, wind_dir = fetch_current_weather(center_lat, center_lon)
        
        if temp is not None:
            df['temperature'] = temp
            df['wind_speed'] = wind_speed
            df['wind_dir'] = wind_dir
            print("Weather successfully integrated.")
        else:
            print("Weather fetching failed. Injecting NaN values instead.")
            df['temperature'] = pd.NA
            df['wind_speed'] = pd.NA
            df['wind_dir'] = pd.NA
        return df
    except Exception as e:
        print(f"Failed to query database: {e}")
        return pd.DataFrame()
                
                
def get_bbox(continent_name='europe'):
    
    bboxes = {
        'europe': (35.0, 75.0, -15.0, 45.0),
        'north_america': (5.0, 85.0, -170.0, -50.0),
        'south_america': (-55.0, 15.0, -85.0, -35.0),
        'asia': (-10.0, 80.0, 40.0, 180.0),
        'australia': (-50.0, -10.0, 110.0, 180.0), # Oceania
        'africa': (-35.0, 40.0, -20.0, 55.0),
        'global': (-90.0, 90.0, -180.0, 180.0)
    }
    
    clean_string = str(continent_name).strip().lower().replace(' ', '_')
    
    if clean_string not in bboxes:
        print("Invalid continent. Defaulting to Europe")
        return bboxes['europe']
    
    return bboxes[clean_string]