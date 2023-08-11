import os
import sqlite3
import numpy as np
from lib import db_colmap


def AssignCameras(database_path, N_CAMERAS):
    if os.path.exists(database_path):
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        cursor.execute("SELECT image_id, camera_id, name FROM images;")

        for row in cursor.fetchall():
            image_id, camera_id, image_name = row
            camera, _ = image_name.split("/", 1)
            camera = camera[3:]
            cursor.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", (str(int(camera)+1), image_id))

        cursor.execute("SELECT camera_id FROM cameras;")
        for row in cursor.fetchall():
            camera_id = row
            cursor.execute("DELETE FROM cameras WHERE camera_id > ?", (str(N_CAMERAS)))

        connection.commit()
        cursor.close()
        connection.close()

    else:
        print("Database does not exist")
        quit()


def CreateCameras(cam_calib, database_path):
    db = db_colmap.COLMAPDatabase.connect(database_path)
    db.create_tables()

    for c in range(len(cam_calib)):
        cam_type, width, height, dist_par = cam_calib[c]["cam_type"], cam_calib[c]["width"], cam_calib[c]["height"], cam_calib[c]["dist_par"]
        model = int(cam_type)
        width, height = int(width), int(height)
        float_values = [np.float32(value) for value in dist_par.split(',')]
        params = tuple(float_values)
        db.add_camera(model, width, height, params, 1)

    db.commit()