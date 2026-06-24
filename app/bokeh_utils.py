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


def dashboard_fig(df=None):
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
    ranges = CONTINENT_RANGES.get('continent', 'europe')
    
    source = AjaxDataSource(
        data_url=f'/api/bokeh_data_ten_mins?continent={continent}',
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
