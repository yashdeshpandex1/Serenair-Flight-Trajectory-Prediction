import numpy as np
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def build_dashboard_figures(df=None):

    # --- KPI ---
    total = df['icao24'].nunique() if df is not None and not df.empty else 1452
    fig_kpi = go.Figure(go.Indicator(
        mode="number", value=total,
        number={'font': {'color': '#00FFCC', 'size': 60}, 'valueformat': ','}
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
            colors=['#00FFCC', '#00b38f', '#ff4d79', '#FF0044', '#b30030', '#888888', '#444444'],
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
        x=cats, y=counts, marker_color='#00FFCC',
        text=counts, textposition='auto', textfont=dict(color='white')
    ))
    fig_bar.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, color='#888888'),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', zeroline=False)
    )

    # --- Wind trend: avg altitude binned by minute ---
    if df is not None and not df.empty and 'timestamp' in df.columns and 'baro_altitude' in df.columns:
        df_s = df.sort_values('timestamp').copy()
        df_s['bin'] = df_s['timestamp'].dt.floor('1min')
        wb = df_s.groupby('bin')['baro_altitude'].mean().tail(12)
        x_w = [t.strftime('%H:%M') for t in wb.index]
        y_w = wb.values.tolist()
    else:
        x_w = [str(i) for i in range(12, 0, -1)]
        y_w = np.abs(
            np.random.normal(loc=15, scale=3, size=12) +
            np.sin(np.linspace(0, 3.14, 12)) * 8
        ).tolist()
    fig_wind = go.Figure(go.Scatter(
        x=x_w, y=y_w,
        fill='tozeroy', mode='lines+markers',
        line=dict(color='#FF0044', width=3, shape='spline'),
        marker=dict(size=6, color='#FF0044', symbol='circle'),
        fillcolor='rgba(255, 0, 68, 0.15)'
    ))
    fig_wind.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, color='#888888', tickfont=dict(size=9),
                title='Time', title_font=dict(size=10),
                nticks=6),          # <-- only show 6 ticks max
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888',
                zeroline=False, title='Altitude (m)', title_font=dict(size=10))
    )

    # --- Altitude histogram ---
    alt_col = next(
        (c for c in ['baro_altitude', 'geo_altitude']
         if df is not None and not df.empty and c in df.columns),
        None
    )
    altitudes = (
        df[alt_col].dropna().tolist() if alt_col
        else np.random.normal(loc=10000, scale=2000, size=1452).tolist()
    )
    # Convert to km so axis labels are short clean numbers
    altitudes_km = [a / 1000 for a in altitudes]

    altitudes_km = np.array(altitudes_km, dtype=float)
    altitudes_km = altitudes_km[np.isfinite(altitudes_km)]
    altitudes_km = altitudes_km[(altitudes_km >= 0) & (altitudes_km <= 20)]
    altitudes_km = altitudes_km.tolist()

    if altitudes_km:
        alt_min = min(altitudes_km)
        alt_max = max(altitudes_km)
    else:
        alt_min, alt_max = 0, 14

    span = max(alt_max - alt_min, 1)
    bin_size = round(span / 18, 1) or 0.1  
    alt_min = round(alt_min, 1)

    fig_alt = go.Figure(go.Histogram(
        x=altitudes_km,
        xbins=dict(start=alt_min, end=alt_max + bin_size, size=bin_size),
        marker_color='#FF0044',
        marker_line_color='#0D0D0D',
        marker_line_width=1,
        opacity=0.85
    ))
    fig_alt.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        bargap=0.08,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(showgrid=False, color='#888888',
                   title='Altitude (km)', title_font=dict(size=10),
                   tickformat=',.1f'),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888', zeroline=False)
    )

    # --- Velocity vs Altitude scatter ---
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
        marker=dict(color='#00FFCC', size=5, opacity=0.6)
    ))
    fig_scatter.update_layout(
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='#333333', color='#888888',
                   title='Velocity (m/s)', title_font=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#888888',
                   title='Altitude (m)', title_font=dict(size=10))
    )

    def to_dict(fig):
        return json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))

    return {
        'kpi':     to_dict(fig_kpi),
        'region':  to_dict(fig_region),
        'bar':     to_dict(fig_bar),
        'wind':    to_dict(fig_wind),
        'alt':     to_dict(fig_alt),
        'scatter': to_dict(fig_scatter),
    }