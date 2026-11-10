import pycolmap
import sqlite3

# Step 1: Load the reconstruction
reconstruction = pycolmap.Reconstruction("/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/input/")  # should contain cameras.bin, images.bin, points3D.bin
database_path = "/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/database.db"
images_path = "/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/images"
output_path = "/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/outpu2/"

options = pycolmap.IncrementalPipelineOptions()
mapper_options = options.mapper
mapper_options.filter_max_reproj_error=100.0 # 4.0
mapper_options.filter_min_tri_angle=0.1 # 1.5
triangulator_options = options.triangulation
triangulator_options.max_transitivity = 10 # 1
create_max_angle_error=1.0 # 2.0
continue_max_angle_error=1.0 # 2.0
merge_max_reproj_error=1.0 # 4.0
ignore_two_view_tracks=False




## Step 2: Open the database
#conn = sqlite3.connect("path/to/database.db")
#cursor = conn.cursor()
#
## Step 3: Extract matches and keypoints
#def get_matches(image_id1, image_id2):
#    cursor.execute("SELECT rows FROM matches WHERE pair_id=?", (pycolmap.database.pair_id(image_id1, image_id2),))
#    row = cursor.fetchone()
#    if row is None:
#        return []
#    return pycolmap.database.decompress_matches(row[0])
#
## Optional: Load keypoints if needed
#def get_keypoints(image_id):
#    cursor.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id,))
#    row = cursor.fetchone()
#    return pycolmap.database.decompress_keypoints(row[0])
#
## Step 4: Triangulate new points (advanced use)
## pycolmap can triangulate points from registered images and matches
## This requires creating a new Reconstruction and registering cameras + images
#new_reconstruction = pycolmap.Reconstruction()
#
## Copy cameras from original
#for cam_id, cam in reconstruction.cameras.items():
#    new_reconstruction.cameras[cam_id] = cam
#
## Copy registered images (with known poses)
#for image_id, image in reconstruction.images.items():
#    new_image = pycolmap.Image()
#    new_image.name = image.name
#    new_image.camera_id = image.camera_id
#    new_image.T = image.T  # known pose
#    new_image.xys = image.xys
#    new_image.point3D_ids = image.point3D_ids
#    new_reconstruction.images[image_id] = new_image

# Perform triangulation
pycolmap.triangulate_points(
    reconstruction=reconstruction,
    database_path=database_path,
    image_path=images_path,
    output_path=output_path,
    options=options,
)

# Save or inspect results
#reconstruction.write("/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/outpu2/")