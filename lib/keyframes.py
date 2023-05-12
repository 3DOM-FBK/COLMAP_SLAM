import numpy as np


class KeyFrame:
    def __init__(self, image_name, keyframe_id, keyframe_name, camera_id, image_id):
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

    @property
    def image_name(self):
        return self._image_name

    @property
    def image_id(self):
        return self._image_id

    @property
    def keyframe_id(self):
        return self._keyframe_id

    @property
    def keyframe_name(self):
        return self._keyframe_name

    @property
    def camera_id(self):
        return self._camera_id

    @property
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

    @property
    def keyframes(self):
        return self._keyframes

    @property
    def keyframes_names(self):
        return [kf.image_name for kf in self._keyframes]

    @property
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
        for keyframe in self._keyframes:
            if keyframe.keyframe_id == keyframe_id:
                return keyframe
        return None

    def set_keyframe_as_oriented(self, keyframe_id: int) -> None:
        keyframe = self.get_keyframe_by_id(keyframe_id)
        keyframe.oriented = True


if __name__ == "__main__":
    pass
