import numpy as np
from typing import List, Union
from pathlib import Path

# Example
# keyframe._image_name imgs\cam0\1403636871851666432.jpg
# keyframe._image_id 53
# keyframe._keyframe_id 29
# keyframe._keyframe_name 000029.jpg


class KeyFrame:
    def __init__(self, image_name : Union[str, Path], keyframe_id, keyframe_name, camera_id, image_id):
        #for n, char in enumerate(image_name):
        #    if char == '\\':
        #        image_name[n] = '/'
        self._image_name = image_name
        self._image_id = image_id
        self._keyframe_id = keyframe_id
        self._keyframe_name = keyframe_name
        self._camera_id = camera_id

        self._oriented = False
        self.n_keypoints = 0

        # Position
        self.GPSLatitude = "-"
        self.GPSLongitude = "-"
        self.GPSAltitude = "-"
        self.enuX = "-"
        self.enuY = "-"
        self.enuZ = "-"
        self.slamX = "-"
        self.slamY = "-"
        self.slamZ = "-"

        # Position slave cameras
        #self.slave_cameras = {}
        self.slave_cameras_POS = {}

    def image_name(self):
        return self._image_name

    def image_id(self):
        return self._image_id

    def keyframe_id(self):
        return self._keyframe_id

    def keyframe_name(self):
        return self._keyframe_name

    def camera_id(self):
        return self._camera_id

    def oriented(self):
        return self._oriented

    def set_oriented(self):
        self._oriented = True

    def __repr__(self) -> str:
        return f"KeyFrame {self.keyframe_id} {self._keyframe_name}."

    def __eq__(self, o: object) -> bool:
        return self.keyframe_id == o.keyframe_id


class KeyFrameList:
    def __init__(self):
        self._keyframes = []
        self._current_idx = 0

    def __len__(self):
        return len(self._keyframes)

    def __getitem__(self, keyframe_id):
        return self._keyframes[keyframe_id]

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= len(self._keyframes):
            raise StopIteration
        cur = self._current_idx
        self._current_idx += 1
        return self._keyframes[cur]

    def __repr__(self) -> str:
        return f"KeyframeList with {len(self._keyframes)} keyframes."

    def keyframes(self):
        return self._keyframes

    def keyframes_names(self):
        return [kf.image_name for kf in self._keyframes]

    def keyframes_ids(self):
        return [kf.keyframe_id for kf in self._keyframes]

    def add_keyframe(self, keyframe: KeyFrame) -> None:
        self._keyframes.append(keyframe)

    def get_keyframe_by_image_name(self, image_name: str) -> KeyFrame:
        for keyframe in self._keyframes:
            if keyframe.image_name == image_name:
                return keyframe
        return None

    def get_keyframe_by_image_id(self, image_id: int) -> KeyFrame:
        for keyframe in self._keyframes:
            if keyframe.image_id == image_id:
                return keyframe
        return None

    def get_keyframe_by_name(self, keyframe_name: str) -> KeyFrame:
        for keyframe in self._keyframes:
            if keyframe.keyframe_name == keyframe_name:
                return keyframe
        return None

    def get_keyframe_by_id(self, keyframe_id: int) -> KeyFrame:
        #print('keyframe_id', keyframe_id)
        #print('len(self._keyframes)', len(self._keyframes))
        #print('ciao')
        for keyframe in self._keyframes:
            #print('cio')
            #print('keyframe.keyframe_id()', keyframe.keyframe_id())
            if keyframe.keyframe_id() == keyframe_id:
                return keyframe
        return None

    def set_keyframe_as_oriented(self, keyframe_id: int) -> None:
        keyframe = self.get_keyframe_by_id(keyframe_id)
        keyframe.oriented = True


if __name__ == "__main__":
    pass
