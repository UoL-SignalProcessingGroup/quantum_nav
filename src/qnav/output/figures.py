

import plotly.graph_objects as go
import numpy as np

from plotly.subplots import make_subplots
from qnav.output.data import ResultsTable
from pathlib import Path

from qnav.util.transformations import lla2ned_vec


def generate_map_plot(*result_tables: ResultsTable, map_style: str = 'dark') -> go.Figure:

    if len(result_tables) == 0:
        raise ValueError("No results tables given")

    fig = go.Figure()
    titles = []

    max_lat = float('-inf')
    max_lon = float('-inf')
    min_lat = float('inf')
    min_lon = float('inf')

    for results in result_tables:

        if not results.is_data_loaded:
            results.read()

        data = results.get_data()
        lat_records = data['data']['position'][:, 0]
        lon_records = data['data']['position'][:, 1]
        time_records = data['time_steps']
        name_text = data['description']
        titles.append(name_text)

        max_lat = max(np.nanmax(lat_records), max_lat)
        max_lon = max(np.nanmax(lon_records), max_lon)
        min_lat = min(np.nanmin(lat_records), min_lat)
        min_lon = min(np.nanmin(lon_records), min_lon)

        fig.add_trace(go.Scattermap(
            mode='lines',
            lat=lat_records,
            lon=lon_records,
            text=time_records,
            name=name_text,
            showlegend=True,
            hovertemplate="<b>Lat:</b> %{lat}°<br>"
                          "<b>Lon:</b> %{lon}°<br>"
                          "<extra>" + name_text + "<br>t=%{text}s</extra>",
        ))

    centre_lat = (min_lat + max_lat) / 2
    centre_lon = (min_lon + max_lon) / 2

    max_bound = max(max_lon - min_lon, max_lat - min_lat) * 111
    zoom = 12 - np.log(max_bound)

    title_text = "Map Overlay"
    sub_title_text = " vs. ".join(titles)

    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        map={
            'style': map_style,
            'center': {'lat': centre_lat, 'lon': centre_lon},
            'zoom': zoom
        },
        legend={
            'xanchor': "left",
            'yanchor': "top",
            'x': 0.01,
            'y': 0.99,
        },
        title={
            'text': title_text,
            'xanchor': 'center',
            'yanchor': 'top',
            'y': 0.98,
            'x': 0.5,
            'subtitle': {
                'text': sub_title_text,
                'font': {
                    'color': 'gray',
                    'size': 13
                }
            }
        },
        hoverlabel_align='auto',
    )

    return fig


def generate_2d_plot(*result_tables: ResultsTable) -> go.Figure:

    if len(result_tables) == 0:
        raise ValueError("No results tables given")

    fig = go.Figure()
    titles = []

    for results in result_tables:

        if not results.is_data_loaded:
            results.read()

        data = results.get_data()
        x_records = data['data']['position'][:, 1]
        y_records = data['data']['position'][:, 0]
        time_records = data['time_steps']
        name_text = data['description']
        titles.append(name_text)

        fig.add_trace(go.Scatter(
            mode='lines',
            x=x_records,
            y=y_records,
            text=time_records,
            name=name_text,
            showlegend=True,
            hovertemplate="<b>Lat:</b> %{y}°<br>"
                          "<b>Lon:</b> %{x}°<br>"
                          "<b>Alt:</b> %{z}m<br>"
                          "<extra>" + name_text + "<br>t=%{text}s</extra>",
        ))

    margin_size = 100
    title_text = "Position 2D"
    sub_title_text = " vs. ".join(titles)

    fig.update_layout(
        # template="plotly_dark",
        autosize=True,
        margin={
            'l': margin_size,
            't': margin_size,
            'b': margin_size,
            'r': margin_size
        },
        # legend={
        #     'xanchor': "right",
        #     'yanchor': "top",
        #     'x': 0.01,
        #     'y': 0.99,
        # },
        title={
            'text': title_text,
            'xanchor': 'center',
            'yanchor': 'top',
            'y': 0.98,
            'x': 0.5,
            'subtitle': {
                'text': sub_title_text,
                'font': {
                    'color': 'gray',
                    'size': 13
                }
            }
        },
        xaxis={
            'title': 'Longitude (°)'
        },
        yaxis={
           'title': 'Latitude (°)',
           'scaleanchor': 'x',
           'scaleratio':1
        },
        hoverlabel_align='auto'
    )

    return fig


def generate_3d_plot(*result_tables: ResultsTable) -> go.Figure:

    if len(result_tables) == 0:
        raise ValueError("No results tables given")

    fig = go.Figure()
    titles = []

    for results in result_tables:

        if not results.is_data_loaded:
            results.read()

        data = results.get_data()
        x_records = data['data']['position'][:, 1]
        y_records = data['data']['position'][:, 0]
        z_records = data['data']['position'][:, 2]
        time_records = data['time_steps']
        name_text = data['description']
        titles.append(name_text)

        fig.add_trace(go.Scatter3d(
            mode='lines',
            x=x_records,
            y=y_records,
            z=z_records,
            text=time_records,
            name=name_text,
            showlegend=True,
            hovertemplate="<b>Lat:</b> %{y}°<br>"
                          "<b>Lon:</b> %{x}°<br>"
                          "<b>Alt:</b> %{z}m<br>"
                          "<extra>" + name_text + "<br>t=%{text}s</extra>",
        ))

    title_text = "Position 3D"
    sub_title_text = " vs. ".join(titles)

    fig.update_layout(
        # template="plotly_dark",
        autosize=True,
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        legend={
            'xanchor': "right",
            'yanchor': "top",
            'x': 0.01,
            'y': 0.99,
        },
        title={
            'text': title_text,
            'xanchor': 'center',
            'yanchor': 'top',
            'y': 0.98,
            'x': 0.5,
            'subtitle': {
                'text': sub_title_text,
                'font': {
                    'color': 'gray',
                    'size': 13
                }
            }
        },
        scene={
            'xaxis_title': 'Longitude (°)',
            'yaxis_title': 'Latitude (°)',
            'zaxis_title': 'Altitude (m)'
        },
        hoverlabel_align='auto'
    )

    return fig



def generate_compare_plot(*result_tables: ResultsTable,
                          field_id: str, row_names: list[str],
                          units: list[str] = None) -> go.Figure:

    if len(result_tables) == 0:
        raise ValueError("No results tables given")

    num_rows = len(row_names)
    fig = go.Figure()
    titles = []

    if units is None:
        units = [""] * num_rows

    for results in result_tables:
        if not results.is_data_loaded:
            results.read()

        data = results.get_data()
        records = data['data'][field_id]
        time_records = data['time_steps']
        name_text = data['description']
        titles.append(name_text)

        if num_rows == 1:
            records = records[:, None]

        for i in range(num_rows):
            fig.add_trace(go.Scatter(
                mode='lines',
                x=time_records,
                y=records[:, i],
                yaxis=f"y{i + 1}",
                name=f"{name_text} {row_names[i]}",
                showlegend=True,
                hovertemplate=f"%{{y}} {units[i]}"
                              f"<extra>{name_text}</extra>",
            ))

    sub_title_text = " vs. ".join(titles)

    add_layout = {}
    for i in range(len(row_names)):
        add_layout[f"yaxis{i + 1}"] = {
            'title': f"{row_names[i]} ({units[i]})",
            'exponentformat': 'e'
        }

    fig.update_layout(
        autosize=True,
        title={
            'text': field_id.title(),
            'xanchor': 'center',
            'yanchor': 'top',
            'y': 0.98,
            'x': 0.5,
            'subtitle': {
                'text': sub_title_text,
                'font': {
                    'color': 'gray',
                    'size': 13
                }
            }
        },
        xaxis={
            'title': 'Time (s)'
        },
        **add_layout,
        hoversubplots="axis",
        hovermode="x",
        grid={
            'rows': num_rows,
            'columns': 1
        },
    )

    return fig


def generate_all_compare_plots(table1: ResultsTable, table2: ResultsTable) -> list[go.Figure]:


    all_args = [
        ("position", ["Latitude", "Longitude", "Altitude"], ["°", "°", "m"]),
        ("velocity", ["Along", "Across", "Down"], ["m/s", "m/s", "m/s"]),
        ("acceleration", ["Along", "Across", "Down"], ["m/s²", "m/s²", "m/s²"]),
        ("attitude", ["Heading", "Pitch", "Yaw"], ["°", "°", "°"]),
        ("angle_rates",  ["Heading", "Pitch", "Yaw"], ["°/s", "°/s", "°/s"]),
    ]

    to_return = []
    for args in all_args:
        tmp_args = {'field_id': args[0], 'row_names': args[1], 'units': args[2]}
        to_return.append(generate_compare_plot(table1, table2, **tmp_args))

    return to_return




def _array_to_str_list(array: np.ndarray, fmt: str = ".6f") -> list[str]:
    return [f"{a:{fmt}}" for a in array]

def get_summary_table(data: np.ndarray, row_names: list[str]) -> go.Table:

    table_data = np.atleast_2d(data)
    if table_data.shape[0] != len(row_names):
        raise ValueError("length of row_names and data rows are not equal")

    column_names = ["Axis", "Initial", "Max", "Average", "STD", "Final"]

    last_ind = np.where(~np.isnan(data))[1][-1]
    initial_value = _array_to_str_list(data[:, 0])
    final_value = _array_to_str_list(data[:, last_ind])

    max_value = _array_to_str_list(np.nanmax(np.abs(data), axis=1))
    mean_value = _array_to_str_list(np.nanmean(data, axis=1))
    std_value = _array_to_str_list(np.nanstd(data, axis=1))

    values = np.column_stack((row_names, initial_value, max_value,
                              mean_value, std_value, final_value)).T

    return go.Table(
        header={
            'values': column_names
        },
        cells={
            'values': values
        },
        # cells_font={
        #     'size': 11
        # }
    )

def plot_summary(table_1: ResultsTable, table_2: ResultsTable) -> go.Figure:

    # Ensure all results are loaded:
    if not table_1.is_data_loaded: table_1.read()
    if not table_2.is_data_loaded: table_2.read()

    table_1_data = table_1.get_data()
    table_2_data = table_2.get_data()

    plot_titles = [
        "Position Error (m)",
        "Velocity Error (m/s)",
        "Acceleration Error (m/s²)",
        "Attitude Error (deg)",
        "Angle Rate Error (deg/s)"
    ]

    data_values = [
        "position",
        "velocity",
        "acceleration",
        "attitude",
        "angle_rates"
    ]

    row_names = [
        ["North", "East", "Down"],
        ["Along", "Across", "Down"],
        ["Along", "Across", "Down"],
        ["Heading", "Pitch", "Roll"],
        ["Heading", "Pitch", "Roll"]
    ]

    i = 1
    num_tables = len(data_values)
    fig = make_subplots(rows=num_tables, cols=1, subplot_titles=plot_titles,
                        specs=[[{"type": "table"}]] * num_tables,)

    #
    for item, rows in zip(data_values, row_names):

        data_1 = table_1_data['data'][item]
        data_2 = table_2_data['data'][item]

        if item == "position":
            data_error = lla2ned_vec(data_1, data_2)
        else:
            data_error = data_2 - data_1

        fig.add_trace(get_summary_table(data_error.T, rows), row=i, col=1)
        i += 1

    fig.update_layout(
        # autosize=True,
        # margin={'l': 5, 't': 5, 'b': 5, 'r': 5}

    )

    return fig



def generate_all_figures(estimates: ResultsTable,
                         ground_truth: ResultsTable = None) -> dict:
    """Generate every plot supported by the fields that are available."""

    if not estimates.is_data_loaded:
        estimates.read()
    estimate_fields = set(estimates.get_data()['data'])

    reference_fields = set()
    if ground_truth is not None:
        if not ground_truth.is_data_loaded:
            ground_truth.read()
        reference_fields = set(ground_truth.get_data()['data'])

    all_figures = []
    all_tiles = []
    sizes = []

    position_tables = [estimates]
    if ground_truth is not None and 'position' in reference_fields:
        position_tables.append(ground_truth)

    if 'position' in estimate_fields:
        all_figures.extend([
            generate_map_plot(*position_tables),
            generate_3d_plot(*position_tables),
            generate_2d_plot(*position_tables),
        ])
        all_tiles.extend(['world_map', 'trajectory_3d', 'trajectory_2d'])
        sizes.extend([{'width': 1000, 'height': 1000}] * 3)

    compare_args = [
        ('position', ['Latitude', 'Longitude', 'Altitude'], ['deg', 'deg', 'm']),
        ('velocity', ['Along', 'Across', 'Down'], ['m/s'] * 3),
        ('acceleration', ['Along', 'Across', 'Down'], ['m/s^2'] * 3),
        ('attitude', ['Heading', 'Pitch', 'Roll'], ['deg'] * 3),
        ('angle_rates', ['P', 'Q', 'R'], ['deg/s'] * 3),
    ]
    if ground_truth is not None:
        shared = estimate_fields & reference_fields
        if all(field in shared for field, _, _ in compare_args):
            all_figures.insert(0, plot_summary(estimates, ground_truth))
            all_tiles.insert(0, 'summary')
            sizes.insert(0, {'width': 900, 'height': 1000})
        for field, rows, units in compare_args:
            if field in shared:
                all_figures.append(generate_compare_plot(
                    estimates, ground_truth, field_id=field,
                    row_names=rows, units=units))
                all_tiles.append(field)
                sizes.append({'width': 1600, 'height': 900})

    # a4_width: int = 2480
    # a4_height: int = 3508
    #
    # sizes = [
    #     {'width': 2480, 'height': 3508},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    #     {'width': 3508, 'height': 2480},
    # ]

    ideal_sizes = {
        'figures': all_figures,
        'titles': all_tiles,
        'sizes': sizes
    }

    return ideal_sizes


if __name__ == '__main__':

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'output'

    estimates_table = ResultsTable(data_dir / "estimation.qnr")
    true_data_table = ResultsTable(data_dir / "ground_truth.qnr")

    # figures = generate_all_compare_plots(estimates_table, true_data_table)

    # figure = generate_map_plot(estimates_table, true_data_table)
    # figure = generate_3d_plot(estimates_table, true_data_table)
    # figure = generate_2d_plot(estimates_table, true_data_table)
    # figure = generate_compare_plot(estimates_table, true_data_table,
    #                                field_id="position",
    #                                row_names=["Latitude", "Longitude", "Altitude"],
    #                                units=["°", "°", "m"])
    # figure = generate_3_plots(estimates_table, true_data_table)
    figure = plot_summary(estimates_table, true_data_table)

    figure.write_image("TEST3.png", scale=3, width=900, height=1000)

    #
    # figure.show(renderer="browser", config= {'displaylogo': False})

    # fg = FigureGenerator(estimates_table, true_data_table)

    # estimates_table.

    # loaded_data =

    # plot_world()

    # lat = np.array([52, 53])
    # lon = np.array([-2, -3])
    # fig = px.line_map(lat=lat, lon=lon)
    #
    # # dt = {
    # #     'lat': np.array([[52, 53], [53, 54]]),
    # #     'lon': np.array([[-2, -3], [-3, -4]])
    # # }
    #
    #
    # # fig = px.line_map(dt, lat="lat", lon="lon")
    # fig.show()
