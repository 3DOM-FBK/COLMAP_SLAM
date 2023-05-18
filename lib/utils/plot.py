import os
import cv2
import pickle
import numpy as np

def make_traj_plot(path_to_kfm_obj : str, path_to_pts_obj : str, width : int, height : int):

    img = np.full((height, width, 3), (255,255,255), dtype=np.uint8)
    X = []
    Y = []

    if os.path.exists(path_to_kfm_obj) and os.path.exists(path_to_pts_obj):
        with open(path_to_kfm_obj, "rb") as f:
            my_list = pickle.load(f)
            for obj in my_list:
                if obj.slamX != "-":
                    X.append(float(obj.slamX))
                if obj.slamY != "-":
                    Y.append(float(obj.slamY))

        MAX_X = max(np.abs(X))
        MAX_Y = max(np.abs(Y))
        if MAX_X > MAX_Y:
            MAX = MAX_X
            buffer = width
        else:
            MAX = MAX_Y
            buffer = height

        X = (np.array(X))/MAX*(buffer/2-100) + int(width/2)
        Y = (np.array(Y))/MAX*(buffer/2-100) + int(height/2)

        with open(path_to_pts_obj, "rb") as f:
            points = pickle.load(f)
            for r in range(points.shape[0]):
                p = points[r,:]
                x = p[0]/MAX*(buffer/2-1) + int(width/2)
                y = p[1]/MAX*(buffer/2-1) + int(height/2)
                if int(x) < width and int(x) > 0 and int(y) < height and int(y) > 0:
                    img[int(y), int(x)] = (0, 0, 0)
                #cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1, lineType=16)

        cv2.circle(img, (int(X[0]), int(Y[0])), 3, (255, 0, 0), -1, lineType=16)
        
        last_point = (int(X[0]), int(Y[0]))

        for x, y in zip(X[1:], Y[1:]):
            cv2.line(img, last_point, (int(x), int(y)), (0, 255, 0), lineType=16)
            #img[int(y), int(x)] = (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 0, lineType=16)
            last_point = (int(x), int(y))
    
    return img