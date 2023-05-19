import os
import pickle
import time

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import plotly.graph_objs as go
from matplotlib import interactive
from mpl_toolkits import mplot3d
from plotly.subplots import make_subplots


def read_camera_eo(fname: str = "./keyframes.pkl") -> tuple:
    if not os.path.exists(fname):
        return

    X = []
    Y = []
    Z = []
    with open(fname, "rb") as f:
        my_list = pickle.load(f)
        for obj in my_list:
            if obj.slamX != "-":
                X.append(float(obj.slamX))
            if obj.slamY != "-":
                Y.append(float(obj.slamY))
            if obj.slamZ != "-":
                Z.append(float(obj.slamZ))

    return X, Y, Z


def create_plot():
    global camera_pos_trace, camera_data_trace, fig
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title_text="Camera Trajectory",
    )

    # Add the camera position trace to the figure
    camera_pos_trace = go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="red")
    )
    fig.add_trace(camera_pos_trace)

    # Add the camera data trace to the figure
    camera_data_trace = go.Scatter3d(
        x=[0], y=[0], z=[0], mode="lines", line=dict(color="blue", width=2)
    )
    fig.add_trace(camera_data_trace)

    # Set the plot limits and viewpoint
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        ),
        scene_aspectmode="cube",
    )


def update_plot():
    # Read the camera data
    X, Y, Z = read_camera_eo("./keyframes.pkl")

    # Update the camera position trace
    camera_pos_trace.x = [X[-1]]
    camera_pos_trace.y = [Y[-1]]
    camera_pos_trace.z = [Z[-1]]

    # Update the camera data trace
    camera_data_trace.x = X
    camera_data_trace.y = Y
    camera_data_trace.z = Z

    # Set the plot title
    fig.update_layout(title_text=f"Camera Trajectory (n={len(X)})")


if __name__ == "__main__":
    SLEEP = 1
    condition = True

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 8))

    ax.set_title("Trajectory")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    while condition:
        try:
            X, Y, Z = read_camera_eo("./keyframes.pkl")

            ax.cla()
            if X and Y and Z:
                ax.scatter(X, Y, Z, "black")
                MIN = min([min(X), min(Y), min(Z)])
                MAX = max([max(X), max(Y), max(Z)])
                ax.set_xlim([MIN, MAX])
                ax.set_ylim([MIN, MAX])
                ax.set_zlim([MIN, MAX])
            ax.view_init(azim=90, elev=0)
            plt.pause(SLEEP)

            Total_imgs = len(os.listdir("./colmap_imgs"))
            N_oriented_cameras = len(X)

        except:
            time.sleep(SLEEP)

        if not condition:
            break
