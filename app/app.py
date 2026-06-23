from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template
from bokeh_utils import next_instance_trajectory_map
from dashboard import build_dashboard_figures

import os,  sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from workers.bg_operations import trigger_background_worker
from workers.database_conn import get_connection_uri

app = Flask(__name__)
CORS(app)


@app.route('/')
def home_page():
    return render_template('home.html', page_title='Main Page')

@app.route('/api/wakeup', methods=['POST'])
def wake_up():
    return jsonify({'status': 'worker_active', 'message': 'Data collection.'})


@app.route('/trajectories')
def trajectories_page():
    continent = request.args.get('continent', 'europe')
    
    script, map_div = next_instance_trajectory_map(continent)
    
    return render_template('trajectories.html', script=script,
                           div=map_div, continent=continent)


@app.route('/dashboard')
def dashboard_page():
    figures = build_dashboard_figures()
    return render_template('dashboard.html', figures=figures)


if __name__ == '__main__':
    app.run(debug=True, port=5000)