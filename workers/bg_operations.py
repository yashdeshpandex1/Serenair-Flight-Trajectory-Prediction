import threading
from workers.database_conn import get_connection_uri
import time
from datetime import datetime
from workers.fetch_live_data import fetch_live_flights_data
from workers.db_utils import prod_database_ingestion, prune_old_data
import logging

logger = logging.getLogger(__name__)

WAKE_UP = 0
SCRIPT_RUNNING = False
TIMEOUT_SECONDS = 120
FREQUENCY = 10
worker_lock = threading.Lock()


def start_background_data_collection():
    global SCRIPT_RUNNING, WAKE_UP
    logger.info("Background data collector running.")
    conn_string = get_connection_uri()
    try:
        while True:
            if time.time() - WAKE_UP > TIMEOUT_SECONDS:
                logger.info("No active users, shutting down collector.")
                break

            now = datetime.now()
            df_flights = fetch_live_flights_data()

            if not df_flights.empty:
                prod_database_ingestion(df_flights, conn_string)
                prune_old_data(conn_string, tolerance_mins=15)
                logger.info(f"[{now.strftime('%H:%M:%S')}] Saved {len(df_flights)} flights globally.")
            else:
                logger.info(f"[{now.strftime('%H:%M:%S')}] No active trajectories found.")

            time.sleep(FREQUENCY)

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")

    finally:
        with worker_lock:
            SCRIPT_RUNNING = False
            logger.info("Background worker stopped.")


def trigger_background_worker():
    global WAKE_UP, SCRIPT_RUNNING

    WAKE_UP = time.time()

    with worker_lock:
        if not SCRIPT_RUNNING:
            thread = threading.Thread(target=start_background_data_collection, daemon=True)
            try:
                thread.start()
                SCRIPT_RUNNING = True
                logger.info("Someone entered! Starting the background data collector.")
            except Exception as e:
                logger.error(f"Failed to start background worker: {e}")