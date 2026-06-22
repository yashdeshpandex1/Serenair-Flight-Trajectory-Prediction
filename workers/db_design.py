import psycopg
from database_conn import get_connection_uri

def create_tables():
    conn_string =  get_connection_uri()
    
    create_aircraft_table = """
    CREATE TABLE IF NOT EXISTS aircraft (
        icao24 CHAR(6) PRIMARY KEY,
        origin_country VARCHAR(100),
        category SMALLINT
    );
    """
    
    create_aircraft_states = """
    CREATE TABLE IF NOT EXISTS aircraft_states (
        id BIGSERIAL PRIMARY KEY,
        
        icao24 CHAR(6) REFERENCES aircraft(icao24),
        callsign VARCHAR(8),
        timestamp BIGINT NOT NULL,
        
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        baro_altitude DOUBLE PRECISION,
        
        velocity DOUBLE PRECISION,
        vertical_rate DOUBLE PRECISION,
        true_track DOUBLE PRECISION,
        geo_altitude DOUBLE PRECISION,
        on_ground BOOLEAN,
        
        squawk VARCHAR(10)    
    );
    """
    
    create_indexes = """
    CREATE INDEX IF NOT EXISTS idx_aircraft_states_timestamp ON aircraft_states(timestamp);
    CREATE INDEX IF NOT EXISTS idx_aircraft_states_location ON aircraft_states(latitude, longitude);
    """
    
    queries = [create_aircraft_table, create_aircraft_states, create_indexes]
    
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                print("Connected to Azure, creating tables..")
                
                for i, query in enumerate(queries, 1):
                    cur.execute(query)
                    print(f"Table {i} created succesfully.")
                    
                conn.commit()
    except Exception as e:
        print(f"Failed to execute: {e}")
        
if __name__ == "__main__":
    create_tables()
    
