import os
import cv2
import time
import shutil

from pathlib import Path
from PIL import Image, ImageOps


class FramesReader:
    def __init__(
            self,
            start: int = 0,
            end: int = 100000000,
            show_camera_flow: bool = True,
            ):
        self.start = start
        self.end = end
        self.show_camera_flow = show_camera_flow

    def simulator(
        self,
        frames_dir: Path,
        output_dir: Path = Path("./frames"), 
        ext: str = "jpg", 
        step: int =1, 
        sleep: float = 0.1, 
        equalize: bool = False, 
        n_camera: int = 1,
    ):
        
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for camera in range(n_camera):
            (output_dir / f"cam{camera}").mkdir(parents=True, exist_ok=True)

        master_camera_imgs = os.listdir(frames_dir / "cam0/data")
        for i in range(len(master_camera_imgs)):
            if i < self.start:
                continue
            if i == self.end:
                print(f"Processed {self.end} images. Change the self.end variable in class FramesReader. Quitting.")
                quit()

            if i % step != 0: # Process only every STEP-th image
                continue

            if self.show_camera_flow:
                im_name = Path(master_camera_imgs[i]).name
                image = cv2.imread(str(frames_dir / f"cam0/data" / im_name))
                cv2.imshow("Stream form cam0", image)
                cv2.waitKey(1)

            for c in reversed(range(n_camera)):
                im_name = Path(master_camera_imgs[i]).name
                im = Image.open(frames_dir / f"cam{c}/data" / im_name)
                rgb_im = im.convert("RGB")
                if equalize == True:
                    rgb_im = ImageOps.equalize(rgb_im)
                rgb_im.save(Path(output_dir) / f"cam{c}" / f"{Path(im_name).stem}.{ext}")
            
            time.sleep(sleep)