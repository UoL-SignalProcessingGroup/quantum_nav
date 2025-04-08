
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import numpy as np

from qnav.waypoints.trajectory import Trajectory, Trajectory


def make_3dim_plot(time_steps: np.ndarray, data: np.ndarray, title: str = "data") -> go.Figure:

    fig = make_subplots(rows=3, cols=1)

    sub_fig_1 = go.Scatter(x=time_steps, y=data[:, 0])
    sub_fig_2 = go.Scatter(x=time_steps, y=data[:, 1])
    sub_fig_3 = go.Scatter(x=time_steps, y=data[:, 2])

    fig.add_trace(sub_fig_1, row=1, col=1)
    fig.add_trace(sub_fig_2, row=2, col=1)
    fig.add_trace(sub_fig_3, row=3, col=1)

    fig.update_layout(title_text=title)
    return fig



def plot_trajectory(trajectory: Trajectory, time_steps: np.ndarray):

    to_plot = trajectory.get_records(time_steps)

    plots = [
        make_3dim_plot(time_steps, to_plot['position'], "Position"),
        make_3dim_plot(time_steps, to_plot['velocity'], "Velocity"),
        make_3dim_plot(time_steps, to_plot['acceleration'], "Acceleration"),
        make_3dim_plot(time_steps, to_plot['attitude'], "Attitude"),
        make_3dim_plot(time_steps, to_plot['angle_rates'], "Angle Rates")
    ]

    for plot in plots:
        plot.show(renderer="browser", config= {'displaylogo': False})


