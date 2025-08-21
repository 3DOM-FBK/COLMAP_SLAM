# Improving BA

`src/odometry/custom_bundle_adjustment.py`
```
 def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
+    #print(ba_options); print(ba_config)
+    #ba_options.use_gpu = True
+    #ba_options.min_num_images_gpu_solver = 5
```

`src/odometry/odometry.py` and `src/odometry/custom_incremental_pipeline.py`
Added `self.run_BA` to run BA only when the last image of the rig is added