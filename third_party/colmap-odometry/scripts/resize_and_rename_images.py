import os
import cv2
import shutil
from pathlib import Path

#input_folder = Path(r"D:\Ahmad_Sordine\originala_data\793-cam4")
#out_folder = Path(r"D:\Ahmad_Sordine\originala_data\downsampled\cam4")

#for i,img in enumerate(os.listdir(input_folder)):
#    name = f"{i:06d}.jpg"
#    img = cv2.imread(str(input_folder / img))
#    resized = cv2.resize(img, (612, 512))
#    cv2.imwrite(str(out_folder / name), resized)

## Resize and rename
#for i,img in enumerate(os.listdir(input_folder)):
#    _, _, _, name = img.split('_', 3)
#    index = name[5:-4]
#    index = int(index)
#    name = f"{index:06d}.jpg"
#    img = cv2.imread(str(input_folder / img))
#    resized = cv2.resize(img, (612, 512))
#    cv2.imwrite(str(out_folder / name), resized)

## temp
#input_folder = Path(r"D:\Ahmad_Sordine\originala_data\650-cam0")
#out_folder = Path(r"D:\Ahmad_Sordine\originala_data\fullres\cam0")
#d = {}
#with open(r"D:\Ahmad_Sordine\originala_data\problems\original\keyframes.txt", 'r') as f:
#    lines = f.readlines()
#    for line in lines:
#        original, keyframe = line.strip().split(',', 1)
#        d[Path(original).name] = Path(keyframe).name
#
#for i,img in enumerate(os.listdir(input_folder)):
#    _, _, _, name = img.split('_', 3)
#    index = name[5:-4]
#    index = int(index)
#    name = f"{index:06d}.jpg"
#    if name in list(d.keys()):
#        shutil.copyfile(str(input_folder / img), str(out_folder / d[name]))

#workdir =  Path(r"/home/threedom/Desktop/_DATA/Town03NoLight/")
#images_dir = workdir / "original_images"
#images = os.listdir(images_dir)
#for image in images:
#    if image.endswith(".jpg"):
#        name = image.split(".")[0]
#        if name[-1] == "L":
#            shutil.copyfile(str(images_dir / image), str(workdir / "cam0" / (name[:-1] + ".jpg")))
#        if name[-1] == "R":
#            shutil.copyfile(str(images_dir / image), str(workdir / "cam1" / (name[:-1] + ".jpg")))


clahe = cv2.createCLAHE(
    clipLimit=2.0,        # contrast limiting
    tileGridSize=(8, 8)   # size of local regions
)

workdir =  Path(r"/home/threedom/Desktop/_DATA/CARLA2")
images_dir = workdir / "original_images"
images = os.listdir(images_dir)
for image in images:
    if image.endswith(".jpg"):
        name = image.split(".")[0]
        img = cv2.imread(str(images_dir / image), cv2.IMREAD_GRAYSCALE)
        if name[-1] == "L":
            cl = clahe.apply(img)
            cv2.imwrite(str(workdir / "cam0" / (name[:-1] + ".jpg")), cl)
        if name[-1] == "R":
            cl = clahe.apply(img)
            cv2.imwrite(str(workdir / "cam1" / (name[:-1] + ".jpg")), cl)
