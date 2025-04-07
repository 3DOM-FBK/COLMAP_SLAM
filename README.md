# COLMAP_SLAM v2


- REPOSITORY UNDER DEVELOPMENT -


https://github.com/3DOM-FBK/COLMAP_SLAM/assets/93863149/af549427-8e87-445d-92f3-1c14e42b5d5a

COLMAP_SLAM is a Visual-SLAM based on pycolmap and is mainly intended for the development and testing of new SLAM features (deep-learning based tie points and matching, keyframe selection, global optimization, etc).

If interested in the project please contact us, you are free to join.


## Running the code

```
python ./main.py -c ./config/config_euroc.yaml -a ./calibration/calibration_euroc.yaml -w raw_data/
```
In the working directory `raw_data` is expected a folder called `images` containing a folder for each camera, for instance `cam0` and `cam1`.


### Reference

Morelli, L., Ioli, F., Beber, R., Menna, F., Remondino, F. and Vitti, A., 2023. COLMAP-SLAM: A FRAMEWORK FOR VISUAL ODOMETRY. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48, pp.317-324.


### License

The licence does not cover the thirdparty libraries of local features and matchers, and COLMAP (see related GitHub repositories).

MIT License

Copyright (c) 2023 3DOM-FBK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in alls
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.