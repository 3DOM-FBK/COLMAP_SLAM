import os
import subprocess
import configparser

class ColmapAPI:
    '''Manage COLMAP API'''

    def __init__(self, colmap_exe_path: str) -> None:
        """
        Args:
            colmap_exe_path (str): Path to the executable file of COLMAP
        """
        self.colmap_exe = colmap_exe_path
        self.cfg_root_sift_first_loop = configparser.ConfigParser()
        self.cfg_root_sift_first_loop.read('./lib/sift_first_loop.ini', encoding="utf-8")
        self.cfg_root_sift = configparser.ConfigParser()
        self.cfg_root_sift.read('./lib/sift.ini', encoding="utf-8")
        self.cfg_matcher = configparser.ConfigParser()
        self.cfg_matcher.read('./lib/matcher.ini', encoding="utf-8")
        self.cfg_mapper_first_loop = configparser.ConfigParser()
        self.cfg_mapper_first_loop.read('./lib/mapper_first_loop.ini', encoding="utf-8")
        self.cfg_mapper = configparser.ConfigParser()
        self.cfg_mapper.read('./lib/mapper.ini', encoding="utf-8")
    

    def CreateEmptyDatabase(self, database_path: str) -> None:
        '''
        Create an empty SQLite database if it does not exist.
        
        Args:
            database_path (str): Path to the new database to be created
        '''
        if not os.path.exists(database_path):
            subprocess.run(
                [
                    self.colmap_exe,
                    "database_creator",
                    "--database_path",
                    f"{database_path}",
                ],
                stdout=subprocess.DEVNULL,
            )


    def ExtractRootSiftFeatures(self, database_path: str, path_to_images: str, first_loop: bool, max_n_features: int) -> None:
        '''
        Extract RootSift local features and save them into an existing database.

        Args:
            database_path (str): Path to existing dataset
        '''



        if first_loop == True:       
            subprocess.call(
                [
                    self.colmap_exe,
                    "feature_extractor",
                    "--database_path", database_path,
                    "--image_path", path_to_images,

                    "--ImageReader.single_camera",                  str(self.cfg_root_sift_first_loop["ImageReader"]["single_camera"]),
                    "--ImageReader.single_camera_per_folder",       str(self.cfg_root_sift_first_loop["ImageReader"]["single_camera_per_folder"]),
                    "--ImageReader.existing_camera_id",             str(self.cfg_root_sift_first_loop["ImageReader"]["existing_camera_id"]),
                    "--ImageReader.default_focal_length_factor",    str(self.cfg_root_sift_first_loop["ImageReader"]["default_focal_length_factor"]),
                    "--ImageReader.mask_path",                      str(self.cfg_root_sift_first_loop["ImageReader"]["mask_path"]),
                    "--ImageReader.camera_model",                   str(self.cfg_root_sift_first_loop["ImageReader"]["camera_model"]), # Placeholder, it will be substituted with calib from config.ini
                    "--ImageReader.camera_params",                  str(self.cfg_root_sift_first_loop["ImageReader"]["camera_params"]), # Placeholder, it will be substituted with calib from config.ini
                    "--ImageReader.camera_mask_path",               str(self.cfg_root_sift_first_loop["ImageReader"]["camera_mask_path"]),

                    "--SiftExtraction.use_gpu",                     str(self.cfg_root_sift_first_loop["SiftExtraction"]["use_gpu"]),
                    "--SiftExtraction.estimate_affine_shape",       str(self.cfg_root_sift_first_loop["SiftExtraction"]["estimate_affine_shape"]),
                    "--SiftExtraction.upright",                     str(self.cfg_root_sift_first_loop["SiftExtraction"]["upright"]),
                    "--SiftExtraction.domain_size_pooling",         str(self.cfg_root_sift_first_loop["SiftExtraction"]["domain_size_pooling"]),
                    "--SiftExtraction.num_threads",                 str(self.cfg_root_sift_first_loop["SiftExtraction"]["num_threads"]),
                    "--SiftExtraction.max_image_size",              str(self.cfg_root_sift_first_loop["SiftExtraction"]["max_image_size"]),
                    "--SiftExtraction.max_num_features",            str(max_n_features),
                    "--SiftExtraction.first_octave",                str(self.cfg_root_sift_first_loop["SiftExtraction"]["first_octave"]),
                    "--SiftExtraction.num_octaves",                 str(self.cfg_root_sift_first_loop["SiftExtraction"]["num_octaves"]),
                    "--SiftExtraction.octave_resolution",           str(self.cfg_root_sift_first_loop["SiftExtraction"]["octave_resolution"]),
                    "--SiftExtraction.max_num_orientations",        str(self.cfg_root_sift_first_loop["SiftExtraction"]["max_num_orientations"]),
                    "--SiftExtraction.dsp_num_scales",              str(self.cfg_root_sift_first_loop["SiftExtraction"]["dsp_num_scales"]),
                    "--SiftExtraction.peak_threshold",              str(self.cfg_root_sift_first_loop["SiftExtraction"]["peak_threshold"]),
                    "--SiftExtraction.edge_threshold",              str(self.cfg_root_sift_first_loop["SiftExtraction"]["edge_threshold"]),
                    "--SiftExtraction.dsp_min_scale",               str(self.cfg_root_sift_first_loop["SiftExtraction"]["dsp_min_scale"]),
                    "--SiftExtraction.dsp_max_scale",               str(self.cfg_root_sift_first_loop["SiftExtraction"]["dsp_max_scale"]),
                    "--SiftExtraction.gpu_index",                   str(self.cfg_root_sift_first_loop["SiftExtraction"]["gpu_index"]),
                ],
                stdout=subprocess.DEVNULL,
            )

        elif first_loop == False:
            subprocess.call(
                [
                    self.colmap_exe,
                    "feature_extractor",
                    "--database_path", database_path,
                    "--image_path", path_to_images,

                    "--ImageReader.single_camera",                  str(self.cfg_root_sift["ImageReader"]["single_camera"]),
                    "--ImageReader.single_camera_per_folder",       str(self.cfg_root_sift["ImageReader"]["single_camera_per_folder"]),
                    "--ImageReader.existing_camera_id",             str(self.cfg_root_sift["ImageReader"]["existing_camera_id"]),
                    "--ImageReader.default_focal_length_factor",    str(self.cfg_root_sift["ImageReader"]["default_focal_length_factor"]),
                    "--ImageReader.mask_path",                      str(self.cfg_root_sift["ImageReader"]["mask_path"]),
                    "--ImageReader.camera_model",                   str(self.cfg_root_sift["ImageReader"]["camera_model"]),
                    "--ImageReader.camera_params",                  str(self.cfg_root_sift["ImageReader"]["camera_params"]),
                    "--ImageReader.camera_mask_path",               str(self.cfg_root_sift["ImageReader"]["camera_mask_path"]),

                    "--SiftExtraction.use_gpu",                     str(self.cfg_root_sift["SiftExtraction"]["use_gpu"]),
                    "--SiftExtraction.estimate_affine_shape",       str(self.cfg_root_sift["SiftExtraction"]["estimate_affine_shape"]),
                    "--SiftExtraction.upright",                     str(self.cfg_root_sift["SiftExtraction"]["upright"]),
                    "--SiftExtraction.domain_size_pooling",         str(self.cfg_root_sift["SiftExtraction"]["domain_size_pooling"]),
                    "--SiftExtraction.num_threads",                 str(self.cfg_root_sift["SiftExtraction"]["num_threads"]),
                    "--SiftExtraction.max_image_size",              str(self.cfg_root_sift["SiftExtraction"]["max_image_size"]),
                    "--SiftExtraction.max_num_features",            str(max_n_features),
                    "--SiftExtraction.first_octave",                str(self.cfg_root_sift["SiftExtraction"]["first_octave"]),
                    "--SiftExtraction.num_octaves",                 str(self.cfg_root_sift["SiftExtraction"]["num_octaves"]),
                    "--SiftExtraction.octave_resolution",           str(self.cfg_root_sift["SiftExtraction"]["octave_resolution"]),
                    "--SiftExtraction.max_num_orientations",        str(self.cfg_root_sift["SiftExtraction"]["max_num_orientations"]),
                    "--SiftExtraction.dsp_num_scales",              str(self.cfg_root_sift["SiftExtraction"]["dsp_num_scales"]),
                    "--SiftExtraction.peak_threshold",              str(self.cfg_root_sift["SiftExtraction"]["peak_threshold"]),
                    "--SiftExtraction.edge_threshold",              str(self.cfg_root_sift["SiftExtraction"]["edge_threshold"]),
                    "--SiftExtraction.dsp_min_scale",               str(self.cfg_root_sift["SiftExtraction"]["dsp_min_scale"]),
                    "--SiftExtraction.dsp_max_scale",               str(self.cfg_root_sift["SiftExtraction"]["dsp_max_scale"]),
                    "--SiftExtraction.gpu_index",                   str(self.cfg_root_sift["SiftExtraction"]["gpu_index"]),
                ],
                stdout=subprocess.DEVNULL,
            )


    def SequentialMatcher(self, database_path: str, loop_closure: str, overlap: str, vocab_tree: str) -> None:
        subprocess.call(
            [
                self.colmap_exe,
                "sequential_matcher",
                "--database_path", database_path,

                #"--SiftMatching.use_gpu",                                               str(self.cfg_matcher["SiftMatching"]["use_gpu"]),
                #"--SiftMatching.cross_check",                                           str(self.cfg_matcher["SiftMatching"]["cross_check"]),
                #"--SiftMatching.multiple_models",                                       str(self.cfg_matcher["SiftMatching"]["multiple_models"]),
                #"--SiftMatching.guided_matching",                                       str(self.cfg_matcher["SiftMatching"]["guided_matching"]),
                #"--SiftMatching.num_threads",                                           str(self.cfg_matcher["SiftMatching"]["num_threads"]),
                #"--SiftMatching.max_num_matches",                                       str(self.cfg_matcher["SiftMatching"]["max_num_matches"]),
                #"--SiftMatching.max_num_trials",                                        str(self.cfg_matcher["SiftMatching"]["max_num_trials"]),
                #"--SiftMatching.min_num_inliers",                                       str(self.cfg_matcher["SiftMatching"]["min_num_inliers"]),
                "--SiftMatching.max_ratio",                                             str(self.cfg_matcher["SiftMatching"]["max_ratio"]),
                #"--SiftMatching.max_distance",                                          str(self.cfg_matcher["SiftMatching"]["max_distance"]),
                #"--SiftMatching.max_error",                                             str(self.cfg_matcher["SiftMatching"]["max_error"]),
                #"--SiftMatching.confidence",                                            str(self.cfg_matcher["SiftMatching"]["confidence"]),
                #"--SiftMatching.min_inlier_ratio",                                      str(self.cfg_matcher["SiftMatching"]["min_inlier_ratio"]),
                #"--SiftMatching.gpu_index",                                             str(self.cfg_matcher["SiftMatching"]["gpu_index"]),

                "--SequentialMatching.quadratic_overlap",                               str(self.cfg_matcher["SequentialMatching"]["quadratic_overlap"]),
                "--SequentialMatching.loop_detection",                                  loop_closure,
                "--SequentialMatching.overlap",                                         overlap,
                #"--SequentialMatching.loop_detection_period",                           str(self.cfg_matcher["SequentialMatching"]["loop_detection_period"]),
                #"--SequentialMatching.loop_detection_num_images",                       str(self.cfg_matcher["SequentialMatching"]["loop_detection_num_images"]),
                #"--SequentialMatching.loop_detection_num_nearest_neighbors",            str(self.cfg_matcher["SequentialMatching"]["loop_detection_num_nearest_neighbors"]),
                #"--SequentialMatching.loop_detection_num_checks",                       str(self.cfg_matcher["SequentialMatching"]["loop_detection_num_checks"]),
                #"--SequentialMatching.loop_detection_num_images_after_verification",    str(self.cfg_matcher["SequentialMatching"]["loop_detection_num_images_after_verification"]),
                #"--SequentialMatching.loop_detection_max_num_features",                 str(self.cfg_matcher["SequentialMatching"]["loop_detection_max_num_features"]),
                #"--SequentialMatching.vocab_tree_path",                                 vocab_tree,
            ],
            #stdout=subprocess.DEVNULL,
        )


    def Mapper(self, database_path: str, path_to_images: str, input_path: str, output_path: str, first_loop: bool, loop_counter: int) -> None:
        if first_loop == True:
            subprocess.call(
                [
                    str(self.colmap_exe),
                    "mapper",
                    "--image_path", path_to_images,
                    "--database_path", database_path,
                    "--output_path", output_path,

                    "--Mapper.ignore_watermarks",                           str(self.cfg_mapper_first_loop["Mapper"]["ignore_watermarks"]),
                    "--Mapper.multiple_models",                             str(self.cfg_mapper_first_loop["Mapper"]["multiple_models"]),
                    "--Mapper.extract_colors",                              str(self.cfg_mapper_first_loop["Mapper"]["extract_colors"]),
                    "--Mapper.ba_refine_focal_length",                      str(self.cfg_mapper_first_loop["Mapper"]["ba_refine_focal_length"]),
                    "--Mapper.ba_refine_principal_point",                   str(self.cfg_mapper_first_loop["Mapper"]["ba_refine_principal_point"]),
                    "--Mapper.ba_refine_extra_params",                      str(self.cfg_mapper_first_loop["Mapper"]["ba_refine_extra_params"]),
                    "--Mapper.fix_existing_images",                         str(self.cfg_mapper_first_loop["Mapper"]["fix_existing_images"]),
                    "--Mapper.tri_ignore_two_view_tracks",                  str(self.cfg_mapper_first_loop["Mapper"]["tri_ignore_two_view_tracks"]),
                    "--Mapper.min_num_matches",                             str(self.cfg_mapper_first_loop["Mapper"]["min_num_matches"]),
                    "--Mapper.max_num_models",                              str(self.cfg_mapper_first_loop["Mapper"]["max_num_models"]),
                    "--Mapper.max_model_overlap",                           str(self.cfg_mapper_first_loop["Mapper"]["max_model_overlap"]),
                    "--Mapper.min_model_size",                              str(self.cfg_mapper_first_loop["Mapper"]["min_model_size"]),
                    "--Mapper.init_image_id1",                              str(self.cfg_mapper_first_loop["Mapper"]["init_image_id1"]),
                    "--Mapper.init_image_id2",                              str(self.cfg_mapper_first_loop["Mapper"]["init_image_id2"]),
                    "--Mapper.init_num_trials",                             str(self.cfg_mapper_first_loop["Mapper"]["init_num_trials"]),
                    "--Mapper.num_threads",                                 str(self.cfg_mapper_first_loop["Mapper"]["num_threads"]),
                    "--Mapper.ba_min_num_residuals_for_multi_threading",    str(self.cfg_mapper_first_loop["Mapper"]["ba_min_num_residuals_for_multi_threading"]),
                    "--Mapper.ba_local_num_images",                         str(self.cfg_mapper_first_loop["Mapper"]["ba_local_num_images"]),
                    "--Mapper.ba_local_max_num_iterations",                 str(self.cfg_mapper_first_loop["Mapper"]["ba_local_max_num_iterations"]),
                    "--Mapper.ba_global_images_freq",                       str(self.cfg_mapper_first_loop["Mapper"]["ba_global_images_freq"]),
                    "--Mapper.ba_global_points_freq",                       str(self.cfg_mapper_first_loop["Mapper"]["ba_global_points_freq"]),
                    "--Mapper.ba_global_max_num_iterations",                str(self.cfg_mapper_first_loop["Mapper"]["ba_global_max_num_iterations"]),
                    "--Mapper.ba_global_max_refinements",                   str(self.cfg_mapper_first_loop["Mapper"]["ba_global_max_refinements"]),
                    "--Mapper.ba_local_max_refinements",                    str(self.cfg_mapper_first_loop["Mapper"]["ba_local_max_refinements"]),
                    "--Mapper.snapshot_images_freq",                        str(self.cfg_mapper_first_loop["Mapper"]["snapshot_images_freq"]),
                    "--Mapper.init_min_num_inliers",                        str(self.cfg_mapper_first_loop["Mapper"]["init_min_num_inliers"]),
                    "--Mapper.init_max_reg_trials",                         str(self.cfg_mapper_first_loop["Mapper"]["init_max_reg_trials"]),
                    "--Mapper.abs_pose_min_num_inliers",                    str(self.cfg_mapper_first_loop["Mapper"]["abs_pose_min_num_inliers"]),
                    "--Mapper.max_reg_trials",                              str(self.cfg_mapper_first_loop["Mapper"]["max_reg_trials"]),
                    "--Mapper.tri_max_transitivity",                        str(self.cfg_mapper_first_loop["Mapper"]["tri_max_transitivity"]),
                    "--Mapper.tri_complete_max_transitivity",               str(self.cfg_mapper_first_loop["Mapper"]["tri_complete_max_transitivity"]),
                    "--Mapper.tri_re_max_trials",                           str(self.cfg_mapper_first_loop["Mapper"]["tri_re_max_trials"]),
                    "--Mapper.min_focal_length_ratio",                      str(self.cfg_mapper_first_loop["Mapper"]["min_focal_length_ratio"]),
                    "--Mapper.max_focal_length_ratio",                      str(self.cfg_mapper_first_loop["Mapper"]["max_focal_length_ratio"]),
                    "--Mapper.max_extra_param",                             str(self.cfg_mapper_first_loop["Mapper"]["max_extra_param"]),
                    "--Mapper.ba_global_images_ratio",                      str(self.cfg_mapper_first_loop["Mapper"]["ba_global_images_ratio"]),
                    "--Mapper.ba_global_points_ratio",                      str(self.cfg_mapper_first_loop["Mapper"]["ba_global_points_ratio"]),
                    "--Mapper.ba_global_max_refinement_change",             str(self.cfg_mapper_first_loop["Mapper"]["ba_global_max_refinement_change"]),
                    "--Mapper.ba_local_max_refinement_change",              str(self.cfg_mapper_first_loop["Mapper"]["ba_local_max_refinement_change"]),
                    "--Mapper.init_max_error",                              str(self.cfg_mapper_first_loop["Mapper"]["init_max_error"]),
                    "--Mapper.init_max_forward_motion",                     str(self.cfg_mapper_first_loop["Mapper"]["init_max_forward_motion"]),
                    "--Mapper.init_min_tri_angle",                          str(self.cfg_mapper_first_loop["Mapper"]["init_min_tri_angle"]),
                    "--Mapper.abs_pose_max_error",                          str(self.cfg_mapper_first_loop["Mapper"]["abs_pose_max_error"]),
                    "--Mapper.abs_pose_min_inlier_ratio",                   str(self.cfg_mapper_first_loop["Mapper"]["abs_pose_min_inlier_ratio"]),
                    "--Mapper.filter_max_reproj_error",                     str(self.cfg_mapper_first_loop["Mapper"]["filter_max_reproj_error"]),
                    "--Mapper.filter_min_tri_angle",                        str(self.cfg_mapper_first_loop["Mapper"]["filter_min_tri_angle"]),
                    "--Mapper.tri_create_max_angle_error",                  str(self.cfg_mapper_first_loop["Mapper"]["tri_create_max_angle_error"]),
                    "--Mapper.tri_continue_max_angle_error",                str(self.cfg_mapper_first_loop["Mapper"]["tri_continue_max_angle_error"]),
                    "--Mapper.tri_merge_max_reproj_error",                  str(self.cfg_mapper_first_loop["Mapper"]["tri_merge_max_reproj_error"]),
                    "--Mapper.tri_complete_max_reproj_error",               str(self.cfg_mapper_first_loop["Mapper"]["tri_complete_max_reproj_error"]),
                    "--Mapper.tri_re_max_angle_error",                      str(self.cfg_mapper_first_loop["Mapper"]["tri_re_max_angle_error"]),
                    "--Mapper.tri_re_min_ratio",                            str(self.cfg_mapper_first_loop["Mapper"]["tri_re_min_ratio"]),
                    "--Mapper.tri_min_angle",                               str(self.cfg_mapper_first_loop["Mapper"]["tri_min_angle"]),
                    "--Mapper.snapshot_path",                               str(self.cfg_mapper_first_loop["Mapper"]["snapshot_path"]),
                ],
                stdout=subprocess.DEVNULL,
            )

        elif first_loop == False:

            subprocess.call(
                [
                    str(self.colmap_exe),
                    "image_registrator",
                    "--database_path", database_path,
                    "--input_path", input_path / "0",
                    "--output_path", output_path / "0",

                    "--Mapper.ignore_watermarks",                           str(self.cfg_mapper["Mapper"]["ignore_watermarks"]),
                    "--Mapper.multiple_models",                             str(self.cfg_mapper["Mapper"]["multiple_models"]),
                    "--Mapper.extract_colors",                              str(self.cfg_mapper["Mapper"]["extract_colors"]),
                    "--Mapper.ba_refine_focal_length",                      str(self.cfg_mapper["Mapper"]["ba_refine_focal_length"]),
                    "--Mapper.ba_refine_principal_point",                   str(self.cfg_mapper["Mapper"]["ba_refine_principal_point"]),
                    "--Mapper.ba_refine_extra_params",                      str(self.cfg_mapper["Mapper"]["ba_refine_extra_params"]),
                    "--Mapper.fix_existing_images",                         str(self.cfg_mapper["Mapper"]["fix_existing_images"]),
                    "--Mapper.tri_ignore_two_view_tracks",                  str(self.cfg_mapper["Mapper"]["tri_ignore_two_view_tracks"]),
                    "--Mapper.min_num_matches",                             str(self.cfg_mapper["Mapper"]["min_num_matches"]),
                    "--Mapper.max_num_models",                              str(self.cfg_mapper["Mapper"]["max_num_models"]),
                    "--Mapper.max_model_overlap",                           str(self.cfg_mapper["Mapper"]["max_model_overlap"]),
                    "--Mapper.min_model_size",                              str(self.cfg_mapper["Mapper"]["min_model_size"]),
                    "--Mapper.init_image_id1",                              str(self.cfg_mapper["Mapper"]["init_image_id1"]),
                    "--Mapper.init_image_id2",                              str(self.cfg_mapper["Mapper"]["init_image_id2"]),
                    "--Mapper.init_num_trials",                             str(self.cfg_mapper["Mapper"]["init_num_trials"]),
                    "--Mapper.num_threads",                                 str(self.cfg_mapper["Mapper"]["num_threads"]),
                    "--Mapper.ba_min_num_residuals_for_multi_threading",    str(self.cfg_mapper["Mapper"]["ba_min_num_residuals_for_multi_threading"]),
                    "--Mapper.ba_local_num_images",                         str(self.cfg_mapper["Mapper"]["ba_local_num_images"]),
                    "--Mapper.ba_local_max_num_iterations",                 str(self.cfg_mapper["Mapper"]["ba_local_max_num_iterations"]),
                    "--Mapper.ba_global_images_freq",                       str(self.cfg_mapper["Mapper"]["ba_global_images_freq"]),
                    "--Mapper.ba_global_points_freq",                       str(self.cfg_mapper["Mapper"]["ba_global_points_freq"]),
                    "--Mapper.ba_global_max_num_iterations",                str(self.cfg_mapper["Mapper"]["ba_global_max_num_iterations"]),
                    "--Mapper.ba_global_max_refinements",                   str(self.cfg_mapper["Mapper"]["ba_global_max_refinements"]),
                    "--Mapper.ba_local_max_refinements",                    str(self.cfg_mapper["Mapper"]["ba_local_max_refinements"]),
                    "--Mapper.snapshot_images_freq",                        str(self.cfg_mapper["Mapper"]["snapshot_images_freq"]),
                    "--Mapper.init_min_num_inliers",                        str(self.cfg_mapper["Mapper"]["init_min_num_inliers"]),
                    "--Mapper.init_max_reg_trials",                         str(self.cfg_mapper["Mapper"]["init_max_reg_trials"]),
                    "--Mapper.abs_pose_min_num_inliers",                    str(50),
                    "--Mapper.max_reg_trials",                              str(self.cfg_mapper["Mapper"]["max_reg_trials"]),
                    "--Mapper.tri_max_transitivity",                        str(self.cfg_mapper["Mapper"]["tri_max_transitivity"]),
                    "--Mapper.tri_complete_max_transitivity",               str(self.cfg_mapper["Mapper"]["tri_complete_max_transitivity"]),
                    "--Mapper.tri_re_max_trials",                           str(self.cfg_mapper["Mapper"]["tri_re_max_trials"]),
                    "--Mapper.min_focal_length_ratio",                      str(self.cfg_mapper["Mapper"]["min_focal_length_ratio"]),
                    "--Mapper.max_focal_length_ratio",                      str(self.cfg_mapper["Mapper"]["max_focal_length_ratio"]),
                    "--Mapper.max_extra_param",                             str(self.cfg_mapper["Mapper"]["max_extra_param"]),
                    "--Mapper.ba_global_images_ratio",                      str(self.cfg_mapper["Mapper"]["ba_global_images_ratio"]),
                    "--Mapper.ba_global_points_ratio",                      str(self.cfg_mapper["Mapper"]["ba_global_points_ratio"]),
                    "--Mapper.ba_global_max_refinement_change",             str(self.cfg_mapper["Mapper"]["ba_global_max_refinement_change"]),
                    "--Mapper.ba_local_max_refinement_change",              str(self.cfg_mapper["Mapper"]["ba_local_max_refinement_change"]),
                    "--Mapper.init_max_error",                              str(self.cfg_mapper["Mapper"]["init_max_error"]),
                    "--Mapper.init_max_forward_motion",                     str(self.cfg_mapper["Mapper"]["init_max_forward_motion"]),
                    "--Mapper.init_min_tri_angle",                          str(self.cfg_mapper["Mapper"]["init_min_tri_angle"]),
                    "--Mapper.abs_pose_max_error",                          str(self.cfg_mapper["Mapper"]["abs_pose_max_error"]),
                    "--Mapper.abs_pose_min_inlier_ratio",                   str(self.cfg_mapper["Mapper"]["abs_pose_min_inlier_ratio"]),
                    "--Mapper.filter_max_reproj_error",                     str(self.cfg_mapper["Mapper"]["filter_max_reproj_error"]),
                    "--Mapper.filter_min_tri_angle",                        str(self.cfg_mapper["Mapper"]["filter_min_tri_angle"]),
                    "--Mapper.tri_create_max_angle_error",                  str(self.cfg_mapper["Mapper"]["tri_create_max_angle_error"]),
                    "--Mapper.tri_continue_max_angle_error",                str(self.cfg_mapper["Mapper"]["tri_continue_max_angle_error"]),
                    "--Mapper.tri_merge_max_reproj_error",                  str(self.cfg_mapper["Mapper"]["tri_merge_max_reproj_error"]),
                    "--Mapper.tri_complete_max_reproj_error",               str(self.cfg_mapper["Mapper"]["tri_complete_max_reproj_error"]),
                    "--Mapper.tri_re_max_angle_error",                      str(self.cfg_mapper["Mapper"]["tri_re_max_angle_error"]),
                    "--Mapper.tri_re_min_ratio",                            str(self.cfg_mapper["Mapper"]["tri_re_min_ratio"]),
                    "--Mapper.tri_min_angle",                               str(self.cfg_mapper["Mapper"]["tri_min_angle"]),
                    "--Mapper.snapshot_path",                               str(self.cfg_mapper["Mapper"]["snapshot_path"]),
                ],
                stdout=subprocess.DEVNULL,
            )

            subprocess.call(
                [
                    str(self.colmap_exe),
                    "mapper",
                    "--image_path", path_to_images,
                    "--database_path", database_path,
                    "--input_path", output_path / "0",
                    "--output_path", output_path / "0",

                    "--Mapper.ignore_watermarks",                           str(self.cfg_mapper["Mapper"]["ignore_watermarks"]),
                    "--Mapper.multiple_models",                             str(self.cfg_mapper["Mapper"]["multiple_models"]),
                    "--Mapper.extract_colors",                              str(self.cfg_mapper["Mapper"]["extract_colors"]),
                    "--Mapper.ba_refine_focal_length",                      str(self.cfg_mapper["Mapper"]["ba_refine_focal_length"]),
                    "--Mapper.ba_refine_principal_point",                   str(self.cfg_mapper["Mapper"]["ba_refine_principal_point"]),
                    "--Mapper.ba_refine_extra_params",                      str(self.cfg_mapper["Mapper"]["ba_refine_extra_params"]),
                    "--Mapper.fix_existing_images",                         str(self.cfg_mapper["Mapper"]["fix_existing_images"]),
                    "--Mapper.tri_ignore_two_view_tracks",                  str(self.cfg_mapper["Mapper"]["tri_ignore_two_view_tracks"]),
                    "--Mapper.min_num_matches",                             str(self.cfg_mapper["Mapper"]["min_num_matches"]),
                    "--Mapper.max_num_models",                              str(self.cfg_mapper["Mapper"]["max_num_models"]),
                    "--Mapper.max_model_overlap",                           str(self.cfg_mapper["Mapper"]["max_model_overlap"]),
                    "--Mapper.min_model_size",                              str(self.cfg_mapper["Mapper"]["min_model_size"]),
                    "--Mapper.init_image_id1",                              str(self.cfg_mapper["Mapper"]["init_image_id1"]),
                    "--Mapper.init_image_id2",                              str(self.cfg_mapper["Mapper"]["init_image_id2"]),
                    "--Mapper.init_num_trials",                             str(self.cfg_mapper["Mapper"]["init_num_trials"]),
                    "--Mapper.num_threads",                                 str(self.cfg_mapper["Mapper"]["num_threads"]),
                    "--Mapper.ba_min_num_residuals_for_multi_threading",    str(self.cfg_mapper["Mapper"]["ba_min_num_residuals_for_multi_threading"]),
                    "--Mapper.ba_local_num_images",                         str(self.cfg_mapper["Mapper"]["ba_local_num_images"]),
                    "--Mapper.ba_local_max_num_iterations",                 str(self.cfg_mapper["Mapper"]["ba_local_max_num_iterations"]),
                    "--Mapper.ba_global_images_freq",                       str(self.cfg_mapper["Mapper"]["ba_global_images_freq"]),
                    "--Mapper.ba_global_points_freq",                       str(self.cfg_mapper["Mapper"]["ba_global_points_freq"]),
                    "--Mapper.ba_global_max_num_iterations",                str(self.cfg_mapper["Mapper"]["ba_global_max_num_iterations"]),
                    "--Mapper.ba_global_max_refinements",                   str(self.cfg_mapper["Mapper"]["ba_global_max_refinements"]),
                    "--Mapper.ba_local_max_refinements",                    str(self.cfg_mapper["Mapper"]["ba_local_max_refinements"]),
                    "--Mapper.snapshot_images_freq",                        str(self.cfg_mapper["Mapper"]["snapshot_images_freq"]),
                    "--Mapper.init_min_num_inliers",                        str(self.cfg_mapper["Mapper"]["init_min_num_inliers"]),
                    "--Mapper.init_max_reg_trials",                         str(self.cfg_mapper["Mapper"]["init_max_reg_trials"]),
                    "--Mapper.abs_pose_min_num_inliers",                    str(self.cfg_mapper["Mapper"]["abs_pose_min_num_inliers"]),
                    "--Mapper.max_reg_trials",                              str(self.cfg_mapper["Mapper"]["max_reg_trials"]),
                    "--Mapper.tri_max_transitivity",                        str(self.cfg_mapper["Mapper"]["tri_max_transitivity"]),
                    "--Mapper.tri_complete_max_transitivity",               str(self.cfg_mapper["Mapper"]["tri_complete_max_transitivity"]),
                    "--Mapper.tri_re_max_trials",                           str(self.cfg_mapper["Mapper"]["tri_re_max_trials"]),
                    "--Mapper.min_focal_length_ratio",                      str(self.cfg_mapper["Mapper"]["min_focal_length_ratio"]),
                    "--Mapper.max_focal_length_ratio",                      str(self.cfg_mapper["Mapper"]["max_focal_length_ratio"]),
                    "--Mapper.max_extra_param",                             str(self.cfg_mapper["Mapper"]["max_extra_param"]),
                    "--Mapper.ba_global_images_ratio",                      str(self.cfg_mapper["Mapper"]["ba_global_images_ratio"]),
                    "--Mapper.ba_global_points_ratio",                      str(self.cfg_mapper["Mapper"]["ba_global_points_ratio"]),
                    "--Mapper.ba_global_max_refinement_change",             str(self.cfg_mapper["Mapper"]["ba_global_max_refinement_change"]),
                    "--Mapper.ba_local_max_refinement_change",              str(self.cfg_mapper["Mapper"]["ba_local_max_refinement_change"]),
                    "--Mapper.init_max_error",                              str(self.cfg_mapper["Mapper"]["init_max_error"]),
                    "--Mapper.init_max_forward_motion",                     str(self.cfg_mapper["Mapper"]["init_max_forward_motion"]),
                    "--Mapper.init_min_tri_angle",                          str(self.cfg_mapper["Mapper"]["init_min_tri_angle"]),
                    "--Mapper.abs_pose_max_error",                          str(self.cfg_mapper["Mapper"]["abs_pose_max_error"]),
                    "--Mapper.abs_pose_min_inlier_ratio",                   str(self.cfg_mapper["Mapper"]["abs_pose_min_inlier_ratio"]),
                    "--Mapper.filter_max_reproj_error",                     str(self.cfg_mapper["Mapper"]["filter_max_reproj_error"]),
                    "--Mapper.filter_min_tri_angle",                        str(self.cfg_mapper["Mapper"]["filter_min_tri_angle"]),
                    "--Mapper.tri_create_max_angle_error",                  str(self.cfg_mapper["Mapper"]["tri_create_max_angle_error"]),
                    "--Mapper.tri_continue_max_angle_error",                str(self.cfg_mapper["Mapper"]["tri_continue_max_angle_error"]),
                    "--Mapper.tri_merge_max_reproj_error",                  str(self.cfg_mapper["Mapper"]["tri_merge_max_reproj_error"]),
                    "--Mapper.tri_complete_max_reproj_error",               str(self.cfg_mapper["Mapper"]["tri_complete_max_reproj_error"]),
                    "--Mapper.tri_re_max_angle_error",                      str(self.cfg_mapper["Mapper"]["tri_re_max_angle_error"]),
                    "--Mapper.tri_re_min_ratio",                            str(self.cfg_mapper["Mapper"]["tri_re_min_ratio"]),
                    "--Mapper.tri_min_angle",                               str(self.cfg_mapper["Mapper"]["tri_min_angle"]),
                    "--Mapper.snapshot_path",                               str(self.cfg_mapper["Mapper"]["snapshot_path"]),
                ],
                stdout=subprocess.DEVNULL,
            )

            if loop_counter % 1 == 0:
                subprocess.call(
                    [
                        str(self.colmap_exe),
                        "rig_bundle_adjuster",
                        "--input_path", output_path / "0",
                        "--output_path", output_path / "0",
                        "--rig_config_path", "./cameras.json",

                        "--BundleAdjustment.refine_focal_length", "1",
                        "--BundleAdjustment.refine_extra_params", "1",
                        "--BundleAdjustment.refine_principal_point", "1",
                        "--BundleAdjustment.refine_extrinsics", "1",
                        "--BundleAdjustment.max_num_iterations", "20",
                        "--BundleAdjustment.max_linear_solver_iterations", "20",
                        "--estimate_rig_relative_poses", "1",
                        "--RigBundleAdjustment.refine_relative_poses", "1"
                    ],
                    stdout=subprocess.DEVNULL,
                )

                subprocess.call(
                    [
                        str(self.colmap_exe),
                        "point_filtering",
                        "--input_path", output_path / "0",
                        "--output_path", output_path / "0",
                        "--min_track_len", "5",
                        "--max_reproj_error", "1.5"
                    ],
                    stdout=subprocess.DEVNULL,
                ) # "--min_track_len", "7",