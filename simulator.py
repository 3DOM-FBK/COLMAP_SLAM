import configparser
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

DEBUG = False
END = 100000000  # max number of images to process
SKIP = 1000

def run_simulator(
    input_dir, imgs, output_dir="./imgs", ext="jpg", step=1, sleep=0.1, equalize=False, n_camera=1,
):
    for i in range(len(imgs)):
        if i < SKIP:
            continue
        if i == END:
            logging.info(
                f"Processed {END} images. Change the END variable in simulator.py to process more."
            )
            quit()

        # Process only every STEP-th image
        if i % step != 0:
            continue

        if DEBUG:
            print(f"processing {i}-th img ({i}/{len(imgs)})")

        for c in reversed(range(n_camera)):
            im_name = imgs[i].name
            im = Image.open(input_dir / f"cam{c}/data" / im_name)
            rgb_im = im.convert("RGB")
            if equalize == True:
                rgb_im = ImageOps.equalize(rgb_im)
            rgb_im.save(Path(output_dir) / f"cam{c}" / f"{Path(im_name).stem}.{ext}")
            time.sleep(sleep)

    logging.info("No more images available")


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

    imgs = sorted(Path(input_dir / "cam0/data").glob("*"))
    run_simulator(input_dir, imgs, output_dir, ext, step, sleep, equalize, n_camera)
