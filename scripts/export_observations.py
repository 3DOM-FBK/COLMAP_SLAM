import pickle
import pycolmap
from pathlib import Path

colmap_rec_dir = Path(r"/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/output/")
output_dir = Path(r"/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/output/")

observations = {}
if __name__ == "__main__":
    out_file = output_dir / "observations.pkl"
    reconstruction = pycolmap.Reconstruction(colmap_rec_dir)

    for point3D_id in reconstruction.points3D:
        point3D = reconstruction.points3D[point3D_id]
        X, Y, Z = point3D.xyz
        observations[point3D_id] = {
            "XYZ": (X, Y, Z),
            "track": [],
        }
        track = point3D.track
        for proj in track.elements:
            image_id = proj.image_id
            image_name = reconstruction.images[image_id].name
            point2D_id = proj.point2D_idx
            point2D = reconstruction.images[image_id].points2D[point2D_id]
            x, y = point2D.xy
            observations[point3D_id]["track"].append({
                "image_id": image_id,
                "image_name": image_name,
                "xy": (x, y),
            })

    #print(observations)
    with open(out_file, 'wb') as file:
        pickle.dump(observations, file)


