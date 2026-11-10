import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", type=str, required=True)
parser.add_argument("-k", "--keyframes_log", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
args = parser.parse_args()

images_folder = args.images
keyframes_log = args.keyframes_log
output_file = args.output

with open(keyframes_log, 'r') as f:
    keyframe_names = set(line.strip() for line in f.readlines())

    for keyframe_name in keyframe_names:
        image_path = f"{images_folder}/cam0/{keyframe_name}"
        output_path = f"{output_file}/cam0/{keyframe_name}"
        shutil.copy(image_path, output_path)

        image_path = f"{images_folder}/cam1/{keyframe_name}"
        output_path = f"{output_file}/cam1/{keyframe_name}"
        shutil.copy(image_path, output_path)
