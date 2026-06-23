import threading
from workers.database_conn import get_connection_uri
import time
from datetime import datetime
from workers.database_conn import get_connection_uri
from workers.fetch_live_data import fetch_live_flights_data
from workers.db_utils import prod_database_ingestion, prune_old_data

WAKE_UP = 0
SCRIPT_RUNNING = False
TIMEOUT_SECONDS = 120
FREQUENCY = 10

worker_lock = threading.Lock()


def start_background_data_collection():
    global SCRIPT_RUNNING, WAKE_UP
    print(f"Waking up the background data collector.")
    
    conn_string = get_connection_uri()
    try:
        while True:
            if time.time() - WAKE_UP > TIMEOUT_SECONDS:
                print(f"No active users detected, shutting down the script.")
                break
            
            now = datetime.now()
            df_flights = fetch_live_flights_data()
            
            if not df_flights.empty:
                prod_database_ingestion(df_flights, conn_string)
                prune_old_data(conn_string, tolerance_mins=15)
                print(f"[{now.strftime('%H:%M:%S')}] Saved {len(df_flights)} flights globally.")
            else:
                print(f"[{now.strftime('%H:%M:%S')}] No active trajectories found.")
            
            time.sleep(FREQUENCY)   

    except Exception as e:
        print(f"Ingestion failed: {e}")
        
    finally:
        with worker_lock:
            SCRIPT_RUNNING = False
            print(f"Background worker stopped.")
        
        
        
def trigger_background_worker():
    global WAKE_UP, SCRIPT_RUNNING
    
    WAKE_UP = time.time()
    
    with worker_lock:
        if not SCRIPT_RUNNING:
            SCRIPT_RUNNING = True
            print('Someone entered! Starting the background data collector..')
            thread = threading.Thread(target=start_background_data_collection, daemon=True)
            thread.start()