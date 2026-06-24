from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template
from bokeh_utils import next_instance_trajectory_map, \
    next_ten_mins_trajectory_map
from dashboard import build_dashboard_figures

import os,  sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from workers.bg_operations import trigger_background_worker
from workers.db_utils import fetch_and_integrate_data
from preprocessing.data_prep import prep_live_inference_data

app = Flask(__name__)
CORS(app)

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
    continent = request.args.get('continent', 'europe')
    
    script, map_div = next_instance_trajectory_map(continent)
    
    return render_template('trajectories.html', script=script,
                           div=map_div, continent=continent)
    
@app.route('/api/trajectories', methods=['GET'])
def get_trajectories():
    trigger_background_worker()
    continent = request.args.get('region', 'europe')
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
    
    trajectories = predict(X_tensor, plane_metadata)
    return jsonify({'status': 'success!', 'message': 'collecting 10 timestamps..', 
                    'live_planes': [], 'trajectories': trajectories})


# DASHBOARD
@app.route('/dashboard')
def dashboard_page():
    figures = build_dashboard_figures()
    return render_template('dashboard.html', figures=figures)

@app.route('/api/dashboard-data')
def dashboard_data():
    trigger_background_worker()
    df = fetch_and_integrate_data('global')
    return jsonify(build_dashboard_figures[df if not df.empty else None])


# TEN MINUTES TRAJECTORY PREDICTION
@app.route('/ten-min-trajectories')
def ten_min_trajectories_page():
    trigger_background_worker()
    continent = request.args.get('continent', 'europe')
    
    script, map_div = next_ten_mins_trajectory_map(continent)
    
    return render_template('ten_min_trajectories.html', script=script,
                           div=map_div, continent=continent)


if __name__ == '__main__':
    app.run(debug=True, port=5000)