import os
import cv2
import shutil
from pathlib import Path

input_folder = Path(r"D:\Ahmad_Sordine\originala_data\793-cam4")
out_folder = Path(r"D:\Ahmad_Sordine\originala_data\downsampled\cam4")

#for i,img in enumerate(os.listdir(input_folder)):
#    name = f"{i:06d}.jpg"
#    img = cv2.imread(str(input_folder / img))
#    resized = cv2.resize(img, (612, 512))
#    cv2.imwrite(str(out_folder / name), resized)

for i,img in enumerate(os.listdir(input_folder)):
    _, _, _, name = img.split('_', 3)
    index = name[5:-4]
    index = int(index)
    name = f"{index:06d}.jpg"
    img = cv2.imread(str(input_folder / img))
    resized = cv2.resize(img, (612, 512))
    cv2.imwrite(str(out_folder / name), resized)
