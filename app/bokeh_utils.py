from bokeh.models import AjaxDataSource, WheelZoomTool, HoverTool, Div
from bokeh.plotting import figure
import xyzservices.providers as xyz
from bokeh.layouts import row, column
from bokeh.embed import components
from bokeh.plotting import figure
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import collections, time
import numpy as np
import math
from predict import predict_for_next_instance, \
    predict_for_next_ten_mins
import base64
from sklearn.cluster import DBSCAN

import os,  sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from workers.db_utils import fetch_and_integrate_data
from preprocessing.data_prep import prep_live_inference_data

_sparkline_buffer = collections.deque(maxlen=60)


CONTINENT_RANGES = {
    'europe':        {'x': (-1500000, 4000000),   'y': (4000000, 9000000)},
    'north_america': {'x': (-18000000, -5000000), 'y': (1000000, 12000000)},
    'south_america': {'x': (-10000000, -3500000), 'y': (-7000000, 2000000)},
    'asia':          {'x': (4000000, 20000000),   'y': (-1000000, 10000000)},
    'africa':        {'x': (-2500000, 6000000),   'y': (-4000000, 5000000)},
    'australia':     {'x': (12000000, 20000000),  'y': (-6000000, -1000000)},
    'global':        {'x': (-20000000, 20000000), 'y': (-7000000, 12000000)},
}


def wgs84_to_web_mercator(lon, lat):
    k = 6378137
    x = lon * (k * np.pi / 180.0)
    y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
    return x, y

def get_heading_from_trail(hist):
    if len(hist) >= 2:
        x1, y1 = wgs84_to_web_mercator(hist['longitude'].iloc[-2], hist['latitude'].iloc[-2])
        x2, y2 = wgs84_to_web_mercator(hist['longitude'].iloc[-1], hist['latitude'].iloc[-1])
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return 0.0
        return math.atan2(dy, dx) - (math.pi / 2)
    return 0.0

def get_dynamic_svg_icon(color="#00FFCC"):
    svg_string = f"""
    <svg xmlns="http://www.w3.org/2000/svg" height="32" viewBox="0 0 24 24" width="32" fill="{color}">
        <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 
                 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 
                 19v-5.5l8 2.5z"/>
    </svg>
    """
    encoded_string = base64.b64encode(svg_string.strip().encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded_string}"

ICON_PRED = get_dynamic_svg_icon("#CC0033")
ICON_ACTUAL = get_dynamic_svg_icon("#FFFFFF")


def bokeh_data_helper(continent, task, model, scaler, cap=True):
    df_live = fetch_and_integrate_data(continent)
    MAX_PLANES = 15 if task == 'next_ten_mins' else 10

    data = {
        'current_x': [], 'current_y': [],
        'predicted_x': [], 'predicted_y': [],
        'trail_x': [], 'trail_y': [], 'icao24': [],
        'heading_rad': [], 'icon_pred': [], 'icon_actual': []
    }

    if df_live.empty:
        return data

    X_tensor, plane_metadata = prep_live_inference_data(df_live, window_size=10, task=task)
    df_sorted = df_live.sort_values('timestamp')
    grouped_history = df_sorted.groupby('icao24')

    if X_tensor.nelement() == 0:
        unique_planes = sorted(df_live['icao24'].unique())
        top_n = unique_planes[:MAX_PLANES] if cap else unique_planes
        latest = df_sorted[df_sorted['icao24'].isin(top_n)].groupby('icao24').tail(1)

        for _, row in latest.iterrows():
            cx, cy = wgs84_to_web_mercator(row['longitude'], row['latitude'])
            hist = grouped_history.get_group(row['icao24'])
            tx, ty = wgs84_to_web_mercator(hist['longitude'].values, hist['latitude'].values)

            data['current_x'].append(cx)
            data['current_y'].append(cy)
            data['predicted_x'].append(cx)
            data['predicted_y'].append(cy)
            data['trail_x'].append(tx.tolist())
            data['trail_y'].append(ty.tolist())
            data['icao24'].append(row['icao24'])
            data['heading_rad'].append(get_heading_from_trail(hist))
            data['icon_pred'].append(ICON_PRED)
            data['icon_actual'].append(ICON_ACTUAL)

    else:
        if task == 'next_instance':
            trajectories = predict_for_next_instance(X_tensor, plane_metadata, model, scaler)
        else:
            trajectories = predict_for_next_ten_mins(X_tensor, plane_metadata, model, scaler)

        trajectories = sorted(trajectories, key=lambda x: x['icao24'])
        if cap:
            trajectories = trajectories[:MAX_PLANES]

        for t in trajectories:
            cx, cy = wgs84_to_web_mercator(t['current_position']['lon'], t['current_position']['lat'])
            px, py = wgs84_to_web_mercator(t['prediction_position']['lon'], t['prediction_position']['lat'])

            if t['icao24'] in grouped_history.groups:
                hist = grouped_history.get_group(t['icao24'])
                tx, ty = wgs84_to_web_mercator(hist['longitude'].values, hist['latitude'].values)
                heading = get_heading_from_trail(hist)
            else:
                tx, ty = np.array([cx]), np.array([cy])
                dx, dy = px - cx, py - cy
                heading = math.atan2(dy, dx) - (math.pi / 2) if not (dx == 0 and dy == 0) else 0.0

            data['current_x'].append(cx)
            data['current_y'].append(cy)
            data['predicted_x'].append(px)
            data['predicted_y'].append(py)
            data['trail_x'].append(tx.tolist())
            data['trail_y'].append(ty.tolist())
            data['icao24'].append(t['icao24'])
            data['heading_rad'].append(heading)
            data['icon_pred'].append(ICON_PRED)
            data['icon_actual'].append(ICON_ACTUAL)

    return data


def next_instance_trajectory_map(continent='europe'):
    ranges = CONTINENT_RANGES.get(continent, CONTINENT_RANGES['europe'])
    
    source = AjaxDataSource(
        data_url=f'/api/bokeh_data?continent={continent}',
        polling_interval=10000,
        mode='replace'
    )
    
    # Predicted
    p1 = figure(x_range=ranges['x'], y_range=ranges['y'],
                x_axis_type="mercator", y_axis_type="mercator",
                width=650, height=700, tools="pan,wheel_zoom,reset")
    p1.add_tile(xyz.CartoDB.DarkMatterNoLabels)
    p1.toolbar_location = None
    p1.xgrid.grid_line_alpha = 0.05
    p1.ygrid.grid_line_alpha = 0.05
    p1.toolbar.active_scroll = p1.select_one(WheelZoomTool)
    p1.background_fill_color = '#0D0D0D'
    p1.border_fill_color = '#0D0D0D'
    p1.outline_line_color = None
    p1.multi_line(xs='trail_x', ys='trail_y', source=source, color='#FFFFFF', line_width=1.5, alpha=0.3, line_dash="dashed")
    p1.segment(x0='current_x', y0='current_y', x1='predicted_x', y1='predicted_y',
               source=source, color="#00FFCC", line_width=2.5, alpha=0.9, line_dash="dotted")
    p1.image_url(url='icon_pred', x='current_x', y='current_y',
                 w=24, h=24, w_units='screen', h_units='screen',
                 anchor="center", angle='heading_rad', source=source)
    hover_hitbox_pred = p1.scatter(x='current_x', y='current_y', source=source, size=24, alpha=0, marker='circle')
    p1.add_tools(HoverTool(renderers=[hover_hitbox_pred], tooltips=[("ICAO24", "@icao24")]))
    
    # Actual
    p2 = figure(x_range=p1.x_range, y_range=p1.y_range,
                x_axis_type="mercator", y_axis_type="mercator",
                width=650, height=700, tools="pan,wheel_zoom,reset")
    p2.add_tile(xyz.CartoDB.DarkMatterNoLabels)
    p2.toolbar_location = None
    p2.xgrid.grid_line_alpha = 0.05
    p2.ygrid.grid_line_alpha = 0.05
    p2.toolbar.active_scroll = p2.select_one(WheelZoomTool)
    p2.background_fill_color = '#0D0D0D'
    p2.border_fill_color = '#0D0D0D'
    p2.outline_line_color = None
    p2.multi_line(xs='trail_x', ys='trail_y', source=source, color='#FFFFFF', line_width=2, alpha=0.5, line_dash="dashed")
    p2.image_url(url='icon_actual', x='current_x', y='current_y',
                 w=24, h=24, w_units='screen', h_units='screen',
                 anchor="center", angle='heading_rad', source=source)
    hover_hitbox_actual = p2.scatter(x='current_x', y='current_y', source=source, size=24, alpha=0, marker='circle')
    p2.add_tools(HoverTool(renderers=[hover_hitbox_actual], tooltips=[("ICAO24", "@icao24")]))
    
    label_style = "display:flex; align-items:center; justify-content:center; margin-top:12px; color:white; font-family:'Source Serif 4', serif; font-weight:455; font-size:22px; margin:0;"
    label1 = Div(text=f"<div style='{label_style}'>Predicted</div>", styles={'width': '650px', 'text-align': 'center'})
    label2 = Div(text=f"<div style='{label_style}'>Actual</div>", styles={'width': '650px', 'text-align': 'center'})
    
    layout = row(column(label1, p1), column(label2, p2))
    
    return components(layout)


def build_dashboard_fig(df=None):
    # --- KPI ---
    total = df['icao24'].nunique() if df is not None and not df.empty else 1452
    fig_kpi = go.Figure(go.Indicator(
        mode="number", value=total,
        number={'font': {'color': '#293681', 'size': 60}, 'valueformat': ','}
    ))
    fig_kpi.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=0, r=0, t=10, b=10)
    )

    # --- Regional donut ---
    if df is not None and not df.empty and 'origin_country' in df.columns:
        rc = df.groupby('origin_country')['icao24'].nunique().nlargest(7)
        r_labels, r_values = rc.index.tolist(), rc.values.tolist()
    else:
        r_labels = ['N. America', 'Europe', 'Asia', 'Middle East', 'S. America', 'Oceania', 'Africa']
        r_values = [420, 380, 290, 150, 110, 60, 42]
    fig_region = go.Figure(go.Pie(
        labels=r_labels, values=r_values, hole=0.6,
        marker=dict(
            colors=['#112E81', '#1a3d9e', '#2348b5', "#3358c4", '#4d6fd0', '#4d6fd0', '#9ab2e8'],
            line=dict(color='#0D0D0D', width=2)
        ),
        textinfo='label+percent', textposition='inside',
        insidetextorientation='horizontal',
        textfont=dict(color='white', size=10)
    ))
    fig_region.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=0, r=0, t=10, b=10), showlegend=False
    )

    # --- Bar: velocity buckets ---
    if df is not None and not df.empty and 'velocity' in df.columns:
        v = df['velocity'].dropna()
        cats   = ['Light', 'Small', 'Large', 'Heavy']
        counts = [
            len(v[v < 100]),
            len(v[(v >= 100) & (v < 200)]),
            len(v[(v >= 200) & (v < 300)]),
            len(v[v >= 300])
        ]
    else:
        cats, counts = ['Light', 'Small', 'Large', 'Heavy'], [520, 480, 310, 142]
    fig_bar = go.Figure(go.Bar(
        x=cats, y=counts, marker_color='#293681',
        text=counts, textposition='auto', textfont=dict(color='white')
    ))
    fig_bar.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=55, r=20, t=10, b=30),
        xaxis=dict(showgrid=False, color='#888888'),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', zeroline=False)
    )

    # --- Aircraft count sparkline (replaces flight phase donut) ---
    ts_now = int(time.time())
    count_now = df['icao24'].nunique() if df is not None and not df.empty else 1452
    _sparkline_buffer.append((ts_now, count_now))

    times  = [t for t, _ in _sparkline_buffer]
    counts_spark = [c for _, c in _sparkline_buffer]

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(
        x=times,
        y=counts_spark,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#293681', width=2),
        fillcolor='rgba(41,54,129,0.15)',
    ))
    fig_phase.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=45, r=20, t=10, b=30),
        xaxis=dict(
            showgrid=False, color='#888888', zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', zeroline=False),
    )

    # --- Top 5 origin countries ---
    if df is not None and not df.empty and 'origin_country' in df.columns:
        top5 = df.groupby('origin_country')['icao24'].nunique().nlargest(5)
        c_labels = top5.index.tolist()
        c_values = top5.values.tolist()
    else:
        c_labels = ['United States', 'Germany', 'France', 'China', 'United Kingdom']
        c_values = [420, 210, 185, 160, 140]

    fig_countries = go.Figure(go.Bar(
        x=c_values,
        y=c_labels,
        orientation='h',
        marker_color='#293681',
        text=c_values,
        textposition='auto',
        textfont=dict(color='white', size=10)
    ))
    fig_countries.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=110, r=20, t=10, b=30),
        xaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', zeroline=False),
        yaxis=dict(showgrid=False, color='#888888', autorange='reversed')
    )

    # --- Velocity vs Altitude scatter ---
    alt_col = next(
        (c for c in ['baro_altitude', 'geo_altitude']
         if df is not None and not df.empty and c in df.columns),
        None
    )
    if df is not None and not df.empty and 'velocity' in df.columns and alt_col:
        sample = df[['velocity', alt_col]].dropna().sample(min(500, len(df)))
        vx = sample['velocity'].tolist()
        vy = sample[alt_col].tolist()
    else:
        vx_ = np.random.normal(loc=250, scale=30, size=500)
        vx  = vx_.tolist()
        vy  = (vx_ * 38 + np.random.normal(loc=0, scale=800, size=500)).tolist()
    fig_scatter = go.Figure(go.Scatter(
        x=vx, y=vy, mode='markers',
        marker=dict(color='#4274D9', size=5, opacity=0.6)
    ))
    fig_scatter.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=55, r=20, t=10, b=30),
        xaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', title=None),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', title=None)
    )

    def to_dict(fig):
        return json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))

    return {
        'kpi':       to_dict(fig_kpi),
        'region':    to_dict(fig_region),
        'bar':       to_dict(fig_bar),
        'phase':     to_dict(fig_phase),
        'countries': to_dict(fig_countries),
        'scatter':   to_dict(fig_scatter),
    }
    
    
def next_ten_mins_trajectory_map(continent='europe'):
    ranges = CONTINENT_RANGES.get(continent, CONTINENT_RANGES['europe'])
    
    source = AjaxDataSource(
        data_url=f'/api/bokeh_data_10_mins?continent={continent}',
        polling_interval=10000,
        mode='replace'
    )

    p = figure(x_range=ranges['x'], y_range=ranges['y'],
               x_axis_type="mercator", y_axis_type="mercator",
               width=1300, height=700, tools="pan,wheel_zoom,reset")
    p.add_tile(xyz.CartoDB.DarkMatterNoLabels)
    p.toolbar_location = None
    p.xgrid.grid_line_alpha = 0.05
    p.ygrid.grid_line_alpha = 0.05
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.background_fill_color = '#0D0D0D'
    p.border_fill_color = '#0D0D0D'
    p.outline_line_color = None
    p.multi_line(xs='trail_x', ys='trail_y', source=source,
                 color='#FFFFFF', line_width=1.5, alpha=0.3, line_dash="dashed")
    p.segment(x0='current_x', y0='current_y', x1='predicted_x', y1='predicted_y',
              source=source, color="#CC0033", line_width=2.5, alpha=0.9, line_dash="dotted")
    p.image_url(url='icon_pred', x='current_x', y='current_y',
                w=24, h=24, w_units='screen', h_units='screen',
                anchor="center", angle='heading_rad', source=source)
    hitbox = p.scatter(x='current_x', y='current_y', source=source, size=24, alpha=0, marker='circle')
    p.add_tools(HoverTool(renderers=[hitbox], tooltips=[("ICAO24", "@icao24")]))

    return components(p)


def build_cluster_data(bokeh_data):

    result = {
        'cluster_x': [], 'cluster_y': [], 'cluster_radius': [],
        'cluster_color': [], 'cluster_count': []
    }

    if not bokeh_data['predicted_x']:
        return result

    coords = np.array(list(zip(bokeh_data['predicted_x'], bokeh_data['predicted_y'])))
    db = DBSCAN(eps=100_000, min_samples=4).fit(coords)
    labels = db.labels_

    for label in set(labels.tolist()):
        if label == -1:
            continue
        mask = labels == label
        count = int(mask.sum())

        if count > 500:
            print(f"Label {label}: skipped (mega-cluster, {count} planes)")
            continue

        cluster_points = coords[mask]
        center_x = float(cluster_points[:, 0].mean())
        center_y = float(cluster_points[:, 1].mean())
        distances = np.sqrt(
            (cluster_points[:, 0] - center_x) ** 2 +
            (cluster_points[:, 1] - center_y) ** 2
        )
        radius = float(max(distances.std(), 100_000))

        # Tier coloring
        if count >= 13:
            color = '#8B1A2E'   # red — high density
        elif count >= 7:
            color = '#A67C00'   # yellow — moderate
        else:
            color = '#6B3A7D'  # purple — mild

        result['cluster_x'].append(center_x)
        result['cluster_y'].append(center_y)
        result['cluster_radius'].append(radius)
        result['cluster_color'].append(color)
        result['cluster_count'].append(count)

    return result


def crowd_density_map(continent='europe'):
    ranges = CONTINENT_RANGES.get(continent, CONTINENT_RANGES['europe'])

    source = AjaxDataSource(
        data_url=f'/api/bokeh_data_crowding_clusters?continent={continent}',
        polling_interval=10000,
        mode='replace'
    )

    p = figure(x_range=ranges['x'], y_range=ranges['y'],
               x_axis_type="mercator", y_axis_type="mercator",
               width=1300, height=700, tools="pan,wheel_zoom,reset")
    p.add_tile(xyz.CartoDB.DarkMatterNoLabels)
    p.toolbar_location = None
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.background_fill_color = '#0D0D0D'
    p.border_fill_color = '#0D0D0D'
    p.outline_line_color = None
    p.xgrid.grid_line_alpha = 0.05
    p.ygrid.grid_line_alpha = 0.05

    p.circle(x='cluster_x', y='cluster_y', radius='cluster_radius',
             source=source,
             fill_color='cluster_color', fill_alpha=0.2,
             line_color='cluster_color', line_width=1.5)

    p.add_tools(HoverTool(tooltips=[
        ("Planes in zone", "@cluster_count"),
    ]))

    return components(p)