import cv2
import copy
import time
import configparser
from pathlib import Path
from PIL import Image, ImageOps

DEBUG = False
END = 100000000  # max number of images to process
SKIP = 50 #2000, 1500, 5500


def SortByTimeStamp(original_img_list):
    sorted_img_list = []
    imgs = copy.deepcopy(original_img_list)
    for i in range(len(imgs)):
        im = imgs[i]
        im = Path(im)
        name = str(Path(im.name).stem)
        sec = int(name[:10])
        nano_ = name[10:]
        missing = len(nano_)-9
        for m in range(missing):
            nano_ = nano_ + '0'
        nano = int(nano_)
        sorted_img_list.append((im, sec, nano))
    sorted_img_list.sort(key=lambda x: (x[1], x[2]))
    sorted_img_list = [t[0] for t in sorted_img_list]
    return sorted_img_list

def run_simulator(
    input_dir, imgs, output_dir="./imgs", ext="jpg", step=1, sleep=0.1, equalize=False, n_camera=1,
):

    ## Sort images respect seconds and nanoseconds
    #imgs_to_be_sorted = []
    #for i in range(len(imgs)):
    #    im = imgs[i]
    #    im = Path(im)
    #    name = str(Path(im.name).stem)
    #    #sec = int(name[:-9])
    #    #nano = int(name[-9:])
    #    sec = int(name[:10])
    #    nano_ = name[10:]
    #    missing = len(nano_)-9
    #    for m in range(missing):
    #        nano_ = nano_ + '0'
    #    nano = int(nano_)
    #    imgs_to_be_sorted.append((im, sec, nano))
    #imgs_to_be_sorted.sort(key=lambda x: (x[1], x[2]))
    #imgs = [t[0] for t in imgs_to_be_sorted]

    imgs = SortByTimeStamp(imgs)

    #cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)
    for i in range(len(imgs)):
        if i < SKIP:
            continue
        if i == END:
            print(
                f"Processed {END} images. Change the END variable in simulator.py to process more."
            )
            quit()

        # Process only every STEP-th image
        if i % step != 0:
            continue

        if DEBUG:
            print(f"processing {i}-th img ({i}/{len(imgs)})")

        im_name = imgs[i].name
        image = cv2.imread(str(input_dir / f"cam0/data" / im_name))
        cv2.imshow("Image Viewer", image)
        cv2.waitKey(1)

        for c in reversed(range(n_camera)):
            im_name = imgs[i].name
            #im_name = im_name[8:] # for ant3d
            im = Image.open(input_dir / f"cam{c}/data" / im_name)
            #im = Image.open(input_dir / f"cam{c}/data" / f"ANT3D_{c+1}_{im_name}") # for ant3d
            rgb_im = im.convert("RGB")
            if equalize == True:
                rgb_im = ImageOps.equalize(rgb_im)
            #rgb_im.thumbnail((960, 600), Image.Resampling.LANCZOS) # for ant3d
            rgb_im.save(Path(output_dir) / f"cam{c}" / f"{Path(im_name).stem}.{ext}")

        time.sleep(sleep)

    print("No more images available")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    input_dir = Path(config["DEFAULT"]["SIMULATOR_IMG_DIR"])
    output_dir = config["DEFAULT"]["IMGS_FROM_SERVER"]
    ext = config["DEFAULT"]["IMG_FORMAT"]
    step = int(config["DEFAULT"]["STEP"])
    sleep = float(config["DEFAULT"]["SIMULATOR_SLEEP_TIME"])
    equalize = config["DEFAULT"].getboolean("EQUALIZE")
    n_camera = int(config["CALIBRATION"]["N_CAMERAS"])

    path_to_kfrms_cam0 = Path(input_dir / "cam0/data")
    if not path_to_kfrms_cam0.exists():
        print(
            '\nERROR: Keyframe directory cam0:', 
            Path(input_dir / "cam0/data"), 
            '\nKeframe dir do not exist, check the input path:',
            '\nExpected dir structures:'
            '\ninput_path',
            '\n--> cam0',
            '\n----> data',
            '\n------> 00000.jpg',
            )

    imgs = sorted(Path(input_dir / "cam0/data").glob("*"))
    run_simulator(input_dir, imgs, output_dir, ext, step, sleep, equalize, n_camera)
