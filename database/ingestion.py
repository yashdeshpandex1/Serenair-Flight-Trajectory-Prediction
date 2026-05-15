from database_conn import get_connection_uri 
import pandas as pd
import psycopg
from tqdm import tqdm

def ingest_data():
    """
    """
    conn_string = get_connection_uri() # get a connection uri
    
    df = pd.read_csv("../data/opensky_raw.csv") # read data from csv
    
    # get unique aircrafts for aircraft table
    unique_aircraft = df[['icao24', 'origin_country']].drop_duplicates()
    states_columns = [
        'icao24', 'callsign', 'timestamp', 'latitude', 'longitude', 
        'baro_altitude', 'velocity', 'vertical_rate', 'true_track', 
        'geo_altitude', 'on_ground'
    ]
    chunk_size = 100000
    total_inserted = 0
    
    with open("../data/opensky_raw.csv", 'r') as f:
        total_rows = sum(1 for _ in f) - 1
        
    try: 
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                print("Connected to Azure! Starting ingestion...")
                
                with tqdm(total=total_rows, desc="Uploading to Azure", unit='rows') as pbar:
                    for chunk in pd.read_csv("../data/opensky_raw.csv", chunksize=chunk_size):
                        # 1. Isolate and upsert unique aircraft 
                        unique_aircraft = chunk[['icao24', 'origin_country']].drop_duplicates()
                        insert_aircraft_table = """
                            INSERT INTO aircraft (icao24, origin_country)
                            VALUES (%s, %s)
                            ON CONFLICT (icao24) DO NOTHING;
                        """
                        cur.executemany(insert_aircraft_table, unique_aircraft.values.tolist())
                        
                        # 2. Stream the state vectors
                        chunk = chunk.dropna(subset=['timestamp', 'latitude', 'longitude'])
                        
                        # Force the timestamp to an integer to strip away the .0
                        chunk['timestamp'] = chunk['timestamp'].astype(int)
                        
                        df_states = chunk[states_columns]
                        copy_query = f"""
                            COPY aircraft_states (
                                {', '.join(states_columns)}
                            )
                            FROM STDIN;
                        """
                        
                        with cur.copy(copy_query) as copy:
                            for row in df_states.itertuples(index=False, name=None):
                                copy.write_row(row)
                                
                        conn.commit()
                        
                        rows_processed = len(chunk)
                        total_inserted += rows_processed
                        pbar.update(rows_processed)
            print(f"All {total_inserted} rows securely loaded into Azure db.")
    
    except Exception as e:
        print(f"\nError: {e}")
        
if __name__ == "__main__":
    ingest_data()