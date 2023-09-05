import os
import cv2
import pickle
import numpy as np

def make_traj_plot(keyframes_dict : dict, path_to_pts_obj : str, width : int, height : int):
    PAD = 50
    img = np.full((height, width, 3), (255,255,255), dtype=np.uint8)
    X = []
    Z = []

    cloud_file = open('./cloud.txt', 'w')

    if os.path.exists(path_to_pts_obj):
        #with open(path_to_kfm_obj, "rb") as f:
        #    my_list = pickle.load(f)
        #    
        #    for i in range(len(my_list)):
        #        obj = my_list.__getitem__(i)
        #        if obj.slamX != "-":
        #            X.append(float(obj.slamX))
        #        if obj.slamZ != "-":
        #            Z.append(float(obj.slamZ))
        #            cloud_file.write(f'{float(obj.slamX)}, {float(obj.slamY)}, {float(obj.slamZ)}, 255, 0, 0\n')

        for key in keyframes_dict.keys():
            if keyframes_dict[key]['slamX'] != "-":
                X.append(keyframes_dict[key]['slamX'])
                Z.append(keyframes_dict[key]['slamZ'])
                cloud_file.write(f'{keyframes_dict[key]["slamX"]}, {keyframes_dict[key]["slamX"]}, {keyframes_dict[key]["slamX"]}, 255, 0, 0\n')

        MAX_X = max(np.abs(X))
        MAX_Z = max(np.abs(Z))
        if MAX_X > MAX_Z:
            MAX = MAX_X
            buffer = width
        else:
            MAX = MAX_Z
            buffer = height
        
        try:
            if int(MAX) == 0: # To avoid division by zero
                MAX = 1
                #return img
        except:
            return img 

        X = -(np.array(X))/MAX*(buffer/2-PAD) + int(width/2)
        Z = (np.array(Z))/MAX*(buffer/2-PAD) + int(height/2)

        with open(path_to_pts_obj, "rb") as f:
            points = pickle.load(f)
            for r in range(points.shape[0]):
                p = points[r,:]
                cloud_file.write(f'{p[0]}, {p[1]}, {p[2]}, 0, 0, 0\n')
                x = -p[0]/MAX*(buffer/2-PAD) + int(width/2)
                z = p[2]/MAX*(buffer/2-PAD) + int(height/2)
                if int(x) < width and int(x) > 0 and int(z) < height and int(z) > 0:
                    img[int(z), int(x)] = (0, 0, 0)
                #cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1, lineType=16)

        cv2.circle(img, (int(X[0]), int(Z[0])), 3, (255, 0, 0), -1, lineType=16)
        
        last_point = (int(X[0]), int(Z[0]))

        for x, z in zip(X[1:], Z[1:]):
            cv2.line(img, last_point, (int(x), int(z)), (0, 255, 0), lineType=16)
            #img[int(y), int(x)] = (0, 0, 255)
            cv2.circle(img, (int(x), int(z)), 1, (0, 0, 255), 0, lineType=16)
            last_point = (int(x), int(z))
    
    cloud_file.close()
    
    return img