from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
from flask import render_template
from bokeh_utils import next_instance_trajectory_map, next_ten_mins_trajectory_map, bokeh_data_helper, crowd_density_map, build_cluster_data, build_dashboard_fig
from inference_utils import initialize_inference_engine
import redis
import json

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from workers.bg_operations import trigger_background_worker, set_models
from workers.db_utils import fetch_and_integrate_data
from preprocessing.data_prep import prep_live_inference_data
from predict import predict_for_next_ten_mins, predict_for_next_instance


app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# CACHING
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_client = redis.Redis(host=redis_host, port=6379, db=0)
CACHE_TTL = 15
MAP_COMPONENTS_TTL = 3600 


def get_cached_map_components(continent, map_fn):
    cache_key = f"map_components:{map_fn.__name__}:{continent}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            result = json.loads(cached)
            return result['script'], result['div']
    except Exception as e:
        app.logger.error(f"Redis error: {e}")

    script, div = map_fn(continent)
    try:
        redis_client.set(cache_key, json.dumps({'script': script, 'div': div}), ex=MAP_COMPONENTS_TTL)
    except Exception as e:
        app.logger.error(f"Redis set error: {e}")
    return script, div


def get_cached_map_data(continent, task, model, scaler):
    cache_key = f"bokeh_map:{task}:{continent}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        app.logger.error(f"Redis error: {e}")

    data = bokeh_data_helper(
        continent=continent,
        task=task,
        model=model,
        scaler=scaler
    )
    try:
        redis_client.set(cache_key, json.dumps(data), ex=CACHE_TTL)
    except Exception as e:
        app.logger.error(f"Redis set error: {e}")
    return data



# INITIALISE MODELS AND SCALERS
app.logger.info("Waking up the inference engines..")
try:
    app.logger.info("Loading Next Instance (1-Min) Engine")
    MODEL_NEXT_INSTANCE, SCALER_NEXT_INSTANCE = initialize_inference_engine(
        task='next_instance'
    )
    app.logger.info("Loading Next Ten Mins Engine")
    MODEL_NEXT_TEN_MINS, SCALER_NEXT_TEN_MINS = initialize_inference_engine(
        task='next_ten_mins'
    )
    app.logger.info("All ML Engines initialized successfully!")
    set_models(MODEL_NEXT_INSTANCE, SCALER_NEXT_INSTANCE,
               MODEL_NEXT_TEN_MINS, SCALER_NEXT_TEN_MINS)
except Exception as e:
    app.logger.critical(f"CRITICAL FAILURE during ML initialization: {e}")



def get_cached_cluster_data(continent, cache_ttl=15):
    cache_key = f"crowd_clusters:{continent}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        app.logger.error(f"Redis error: {e}")

    data = bokeh_data_helper(
        continent=continent,
        task='next_ten_mins',
        model=MODEL_NEXT_TEN_MINS,
        scaler=SCALER_NEXT_TEN_MINS,
        cap=False
    )
    cluster_data = build_cluster_data(data)
    try:
        redis_client.set(cache_key, json.dumps(cluster_data), ex=cache_ttl)
    except Exception as e:
        app.logger.error(f"Redis set error: {e}")
    return cluster_data



# HOME PAGE
@app.route('/')
def home_page():
    trigger_background_worker()
    return render_template('home.html', page_title='Main Page')

@app.route('/api/wakeup', methods=['POST'])
def wake_up():
    trigger_background_worker()
    return jsonify({'status': 'worker_active', 'message': 'Data collection.'})


# NEXT INSTANCE TRAJECTORY PREDICTION
@app.route('/trajectories')
def trajectories_page():
    trigger_background_worker()
    continent = request.args.get('continent', 'europe')
    script, map_div = get_cached_map_components(continent, next_instance_trajectory_map)
    return render_template('trajectories.html', script=script,
                           div=map_div, continent=continent)

@app.route('/api/trajectories', methods=['GET'])
def get_trajectories():
    trigger_background_worker()
    continent = request.args.get('continent', 'europe')
    df_live = fetch_and_integrate_data(continent)

    if df_live.empty:
        return jsonify({'status': 'Waking up',
                        'message': 'Waking up the script. Please wait...',
                        'live_planes': [],
                        'trajectories': []}), 202

    X_tensor, plane_metadata = prep_live_inference_data(df_live,
                                                        window_size=10,
                                                        task='next_instance')
    if X_tensor.nelement() == 0:
        latest_positions = df_live.sort_values('timestamp').groupby('icao24').tail(1)
        live_planes = [{'icao24': row['icao24'], 'lat': row['latitude'], 'lon': row['longitude']} for _, row in latest_positions.iterrows()]
        return jsonify({'status': 'calibrating', 'message': 'collecting 10 timestamps..', 'live_planes': live_planes, 'trajectories': []}), 206

    trajectories = predict_for_next_instance(X_tensor, plane_metadata,
                                             model=MODEL_NEXT_INSTANCE,
                                             scaler=SCALER_NEXT_INSTANCE)
    return jsonify({'status': 'success!', 'message': 'collecting 10 timestamps..',
                    'live_planes': [], 'trajectories': trajectories})


# DASHBOARD
@app.route('/dashboard')
def dashboard_page():
    trigger_background_worker()
    cache_key = "dashboard:figures"
    try:
        cached = redis_client.get(cache_key)
        figures = json.loads(cached) if cached else None
    except Exception:
        figures = None
    return render_template('dashboard.html', figures=figures)

@app.route('/api/dashboard-data')
def dashboard_data():
    trigger_background_worker()
    cache_key = "dashboard:figures"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify(json.loads(cached))
    except Exception:
        pass
    figures = build_dashboard_fig(None)
    return jsonify(figures)


# TEN MINUTES TRAJECTORY PREDICTION
@app.route('/ten-mins-trajectories')
def ten_mins_trajectories_page():
    trigger_background_worker()
    continent = request.args.get('continent', 'europe')
    script, map_div = get_cached_map_components(continent, next_ten_mins_trajectory_map)
    return render_template('ten_mins_trajectories.html', script=script,
                           div=map_div, continent=continent)

@app.route('/api/ten-mins-trajectories', methods=['GET'])
def get_ten_mins_trajectories():
    trigger_background_worker()
    continent = request.args.get('continent', 'europe')

    df_live = fetch_and_integrate_data(continent)
    if df_live.empty:
        return jsonify({'status': 'Waking up',
                        'message': 'Waking up the script. Please wait..',
                        'live_planes': [], 'trajectories': []}), 202

    X_tensor, plane_metadata = prep_live_inference_data(df_live,
                                                        window_size=10,
                                                        task='next_ten_mins')
    if X_tensor.nelement() == 0:
        latest_positions = df_live.sort_values('timestamp').groupby('icao24').tail(1)
        live_planes = [{'icao24': row['icao24'], 'lat': row['latitude'],
                        'lon': row['longitude']} for _, row in latest_positions.iterrows()]
        return jsonify({'status': 'calibrating',
                        'message': 'collecting 10 timestamps..',
                        'live_planes': live_planes, 'trajectories': []}), 206

    trajectories = predict_for_next_ten_mins(X_tensor, plane_metadata,
                                             model=MODEL_NEXT_TEN_MINS,
                                             scaler=SCALER_NEXT_TEN_MINS)
    return jsonify({'status': 'success!',
                    'message': '10-Minute Trajectory Prediction active',
                    'live_planes': [], 'trajectories': trajectories}), 200


# CROWD DENSITY PREDICTION
@app.route('/crowd-density-prediction')
def crowd_density_prediction():
    trigger_background_worker()
    continent = request.args.get('continent', 'europe')
    script, map_div = get_cached_map_components(continent, crowd_density_map)
    return render_template('crowd_density_prediction.html', script=script, div=map_div, continent=continent)

@app.route('/api/bokeh_data_crowding_clusters', methods=['GET', 'POST'])
def bokeh_data_crowding_clusters():
    continent = request.args.get('continent', 'global')
    cluster_data = get_cached_cluster_data(continent)
    return jsonify(cluster_data)


# BOKEH DATA
@app.route('/api/bokeh_data', methods=['GET', 'POST'])
def bokeh_data_next_instance():
    continent = request.args.get('continent', 'europe')
    data = get_cached_map_data(
        continent=continent,
        task='next_instance',
        model=MODEL_NEXT_INSTANCE,
        scaler=SCALER_NEXT_INSTANCE
    )
    return jsonify(data)

@app.route('/api/bokeh_data_10_mins', methods=['GET', 'POST'])
def bokeh_data_next_ten_mins():
    continent = request.args.get('continent', 'europe')
    data = get_cached_map_data(
        continent=continent,
        task='next_ten_mins',
        model=MODEL_NEXT_TEN_MINS,
        scaler=SCALER_NEXT_TEN_MINS
    )
    return jsonify(data)

@app.route('/references-and-credits')
def references_and_credits():
    return render_template('credits.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)