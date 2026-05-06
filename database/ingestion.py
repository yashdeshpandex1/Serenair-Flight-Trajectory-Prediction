from database_conn import get_connection_uri 
import pandas as pd
import psycopg

def ingest_data():
    """
    """
    conn_string = get_connection_uri() # get a connection uri
    
    df = pd.read_csv("../data/opensky_raw.csv") # read data from csv
    
    # get unique aircrafts for aircraft table
    unique_aircraft = df[['icao24', 'origin_country']].drop_duplicates()
    states_columns = [
        'icao24', 'callsign', 'time_position', 'latitude', 'longitude', 
        'baro_altitude', 'velocity', 'vertical_rate', 'true_track', 
        'geo_altitude', 'on_ground'
    ]
    df_states = df[states_columns]
    
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                print("Connected to Azure!")
                
                # Insertion in aircraft table
                insert_aircraft_table = """
                    INSERT INTO aircraft (icao24, origin_country)
                    VALUES (%s, %s)
                    ON CONFLICT (icao24) DO NOTHING;
                """
                cur.executemany(insert_aircraft_table, unique_aircraft.values.tolist())
                
                # Insertion in aircraft states table
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
                print("Data successfully loaded.")
        
    except Exception as e:
        print(f" Error: {e}")
        
        
if __name__ == "__main__":
    ingest_data()