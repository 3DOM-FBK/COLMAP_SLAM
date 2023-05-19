import configparser
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

DEBUG = False
END = 1000000  # max number of images to process


def run_simulator(
    imgs, output_dir="./imgs", ext="jpg", step=1, sleep=0.1, equalize=False
):
    for i, img in enumerate(imgs):
        if i == END:
            logging.info(
                f"Processed {END} images. Change the END variable in simulator.py to process more."
            )
            quit()

        # Process only every STEP-th image
        if i % step != 0:
            continue

        if DEBUG:
            print(f"processing {img} ({i}/{len(imgs)})")

        im = Image.open(img)
        rgb_im = im.convert("RGB")
        if equalize == True:
            rgb_im = ImageOps.equalize(rgb_im)
        rgb_im.save(Path(output_dir) / f"{img.stem}.{ext}")
        time.sleep(sleep)

    logging.info("No more images available")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    input_dir = config["DEFAULT"]["SIMULATOR_IMG_DIR"]
    output_dir = config["DEFAULT"]["IMGS_FROM_SERVER"]
    ext = config["DEFAULT"]["IMG_FORMAT"]
    step = int(config["DEFAULT"]["STEP"])
    sleep = float(config["DEFAULT"]["SIMULATOR_SLEEP_TIME"])
    equalize = config["DEFAULT"].getboolean("EQUALIZE")

    imgs = sorted(Path(input_dir).glob("*"))
    run_simulator(imgs, output_dir, ext, step, sleep, equalize)
