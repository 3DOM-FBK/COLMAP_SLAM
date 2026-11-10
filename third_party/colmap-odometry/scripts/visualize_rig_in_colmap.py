import argparse
import sys
import yaml

from pathlib import Path



def load_yaml(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Load a YAML file for the rig visualization.")
    parser.add_argument("config", type=Path, help="Path to the YAML camera_rig.yaml file.")
    parser.add_argument("output", type=Path, help="Path to save poses in COLMAP format.")
    args = parser.parse_args()


    data = load_yaml(args.config)
    print(data)

    camera_file_path = args.output / "cameras.txt"
    images_file_path = args.output / "images.txt"
    points3D_file_path = args.output / "points3D.txt"

    with open(camera_file_path, "w") as cam_file, \
         open(images_file_path, "w") as img_file, \
         open(points3D_file_path, "w") as pts_file:
        
        cam_file.write("1 OPENCV 960 600 370.86733072899773 369.87798294190696 474.36399999999998 302.62099999999998 -0.04409720194307467 -0.0010738029501395044 -0.00012261975723197707 0.0017247579103316375")

        for idx, cam in enumerate(data):
            
            rotation = data[cam]['rotation']
            translation = data[cam]['translation']

            img_file.write(f"{idx+1} {rotation[3]} {rotation[1]} {rotation[2]} {rotation[0]} {translation[0]} {translation[1]} {translation[2]} 1 {cam}\n\n")

if __name__ == "__main__":
    main()