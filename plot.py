# conda activate pillow
# https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib

#mport matplotlib.pyplot as plt
#rom mpl_toolkits import mplot3d
#rom matplotlib import interactive
#mport numpy as np
#mport os
#mport time
#mport pickle
#
#LEEP = 0.2
#ondition = True
#
#
#hile condition == True:
#   if os.path.exists("./keyframes.pkl"):
#       plt.ion()
#       interactive(True)
#       fig = plt.figure()
#       # fig, ax = plt.subplots(3, 1, subplot_kw={'projection' : '3d'}, constrained_layout=True, figsize=(8, 8))
#
#       while condition == True:
#           X = []
#           Y = []
#           Z = []
#           if os.path.exists("./keyframes.pkl"):
#               with open("./keyframes.pkl", "rb") as f:
#                   my_list = pickle.load(f)
#                   for obj in my_list:
#                       if obj.slamX != "-":
#                           X.append(float(obj.slamX))
#                       if obj.slamY != "-":
#                           Y.append(float(obj.slamY))
#                       if obj.slamZ != "-":
#                           Z.append(float(obj.slamZ))
#
#           ax = plt.axes(projection="3d")
#           # mngr = plt.get_current_fig_manager()
#           # mngr.window.setGeometry(50,450,640, 495)
#
#           MIN = min([min(X), min(Y), min(Z)])
#           MAX = max([max(X), max(Y), max(Z)])
#
#           ax.cla()
#           ax.scatter(X, Y, Z, "black")
#           ax.set_title("c")
#           ax.set_xticks([])
#           ax.set_yticks([])
#           ax.set_zticks([])  # ax[2].set_zticks(np.arange(MIN, MAX, (MAX-MIN)/10))
#           ax.view_init(azim=0, elev=90)
#
#           plt.show(block=False)
#           plt.pause(SLEEP)
#           # plt.clf()
#
#           Total_imgs = len(os.listdir("./colmap_imgs"))
#           N_oriented_cameras = len(X)
#
#   else:
#       time.sleep(SLEEP)
#
#
#
#
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive
import numpy as np
import os
import time
import pickle

SLEEP = 1
condition = True

fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
ax.set_title("c")
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
