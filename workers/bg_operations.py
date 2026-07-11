import threading
from .database_conn import get_connection_uri
import time
from datetime import datetime
from .fetch_live_data import fetch_live_flights_data
from .db_utils import prod_database_ingestion, prune_old_data
import logging
import redis
import json
import os

logger = logging.getLogger(__name__)

WAKE_UP = 0
SCRIPT_RUNNING = False
TIMEOUT_SECONDS = 120
FREQUENCY = 10
worker_lock = threading.Lock()

redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)
CACHE_TTL = 30

CONTINENTS = ['europe', 'north_america', 'south_america', 'asia', 'africa', 'australia', 'global']

MODEL_NEXT_INSTANCE = None
SCALER_NEXT_INSTANCE = None
MODEL_NEXT_TEN_MINS = None
SCALER_NEXT_TEN_MINS = None

PRECOMPUTE_EVERY = 1
precompute_counter = 0

def set_models(model_ni, scaler_ni, model_ntm, scaler_ntm):
    global MODEL_NEXT_INSTANCE, SCALER_NEXT_INSTANCE, MODEL_NEXT_TEN_MINS, SCALER_NEXT_TEN_MINS
    MODEL_NEXT_INSTANCE = model_ni
    SCALER_NEXT_INSTANCE = scaler_ni
    MODEL_NEXT_TEN_MINS = model_ntm
    SCALER_NEXT_TEN_MINS = scaler_ntm
    
def precompute_preds(df_global):
    from bokeh_utils import bokeh_data_helper, build_cluster_data
    from workers.db_utils import filter_by_continent
    
    if MODEL_NEXT_INSTANCE is None or MODEL_NEXT_TEN_MINS is None:
        logger.warning("Models not set, skipping precompute")
        return
    
    if df_global.empty:
        logger.warning("No global data available for precompute")
        return
    
    for continent in CONTINENTS:
        df_continent = filter_by_continent(df_global, continent)
        for task, model, scaler in [
            ('next_instance', MODEL_NEXT_INSTANCE, SCALER_NEXT_INSTANCE),
            ('next_ten_mins', MODEL_NEXT_TEN_MINS, SCALER_NEXT_TEN_MINS)
        ]:
            try:
                data = bokeh_data_helper(
                    continent, task, model, scaler, df_override=df_continent)
                cache_key = f"bokeh_map:{task}:{continent}"
                redis_client.set(cache_key, json.dumps(data), ex=CACHE_TTL)
                logger.info(f"Cached {task}:{continent} ({len(data.get('icao24', []))} planes)")
            except Exception as e:
                logger.error(f"Precompute failed for {task}:{continent}: {e}")
        try: 
            cluster_raw_data = bokeh_data_helper(
                continent, 'next_ten_mins', MODEL_NEXT_TEN_MINS, SCALER_NEXT_TEN_MINS,
                cap=False, df_override=df_continent
            )
            cluster_data = build_cluster_data(cluster_raw_data)
            cluster_cache_key = f"crowd_clusters:{continent}"
            redis_client.set(cluster_cache_key, json.dumps(cluster_data), ex=CACHE_TTL)
            logger.info(f"Cached crowd_clusters:{continent} ({len(cluster_data.get('cluster_x', []))} clusters)")
        except Exception as e:
            logger.error(f"Precompute failed for crowd_clusters:{continent}: {e}")

def precompute_dashboard(df_global):
    from bokeh_utils import build_dashboard_fig
    
    if df_global.empty:
        logger.warning("Skipping dashboard precompute: no global data")
        return
    
    try:
        figures = build_dashboard_fig(df_global)
        cache_key = "dashboard:figures"
        redis_client.set(cache_key, json.dumps(figures), ex=CACHE_TTL)
        logger.info("Precomputed dashboard")
    except Exception as e:
        logger.error(f"Dashboard precompute failed: {e}")
    
    
def start_background_data_collection():
    global SCRIPT_RUNNING, WAKE_UP, precompute_counter
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
                precompute_counter += 1
                if precompute_counter >= PRECOMPUTE_EVERY:
                    from workers.db_utils import fetch_global_data
                    df_global = fetch_global_data()
                    precompute_preds(df_global)
                    precompute_dashboard(df_global)
                    precompute_counter = 0
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