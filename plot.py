import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive
import numpy as np
import os
import time
import pickle

SLEEP = 1
condition = True

fig, ax = plt.subplots(subplot_kw={'projection' : '3d'}, figsize=(7, 8))

ax.set_title("Trajectory")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

while condition:
    try:
        with open("./keyframes.pkl", "rb") as f:
            my_list = pickle.load(f)
            X = []
            Y = []
            Z = []
            for obj in my_list:
                if obj.slamX != "-":
                    X.append(float(obj.slamX))
                if obj.slamY != "-":
                    Y.append(float(obj.slamY))
                if obj.slamZ != "-":
                    Z.append(float(obj.slamZ))            

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
