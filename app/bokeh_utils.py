from bokeh.models import AjaxDataSource, WheelZoomTool, HoverTool, Div
from bokeh.plotting import figure
import xyzservices.providers as xyz
from bokeh.layouts import row, column
from bokeh.embed import components
from bokeh.plotting import figure


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
    