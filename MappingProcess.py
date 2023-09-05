# Controllare che i keyframes non vengano sdoppiati

import os
import cv2
import time
import pickle
import logging
import cProfile
import pydegensac
import numpy as np
import subprocess

from lib.colmapAPI import ColmapAPI
from lib.local_features import LocalFeatConfFile, LocalFeatureExtractor
from lib import (
    ExtractCustomFeatures,
    database,
    db_colmap,
    export_cameras,
    import_local_features,
    matcher,
    utils,
    cameras,
)
from lib.keyframes import KeyFrame, KeyFrameList
from lib.keyframe_selection import KeyFrameSelConfFile, KeyFrameSelector
from pathlib import Path
from pyquaternion import Quaternion
from easydict import EasyDict as edict

def MappingProcess(keyframes_list, logger, cfg, newer_imgs, first_colmap_loop, lock, SEQUENTIAL_OVERLAP, adjacency_matrix, keypoints, descriptors, laf, init, SNAPSHOT_DIR, processed_imgs):
    #logger = logging.getLogger(__name__)
    #for i in range(5):
    #    time.sleep(5)
    #    for i in range(10):
    #        logger.info('ciao')
#
    #    #ob = keyframes_list.get_keyframe_by_id(1)
    #    #for k in keyframes_list.keyframes:
    #    print("File list length:",  len(keyframes_list.keyframes()))




    time.sleep(5)
    
    print("Setup local feature to use on keyframes")
    # Setup local feature to use on keyframes
    local_feat_conf = LocalFeatConfFile(cfg)
    local_feat_extractor = LocalFeatureExtractor(
        cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
    )

    print("Initialize COLMAP API")
    # Initialize COLMAP API
    colmap = ColmapAPI(str(cfg.COLMAP_EXE_PATH))



    while True:

        print('Inside while')
        print('len()', len(keyframes_list.keyframes()))
        print(newer_imgs)
        if len(keyframes_list.keyframes()) == 10:
            with lock:
                keyframes_list.add_keyframe(KeyFrame('5','5','5','5','5'))
                c = keyframes_list.get_keyframe_by_name('5')
                c.slamY = 7
                newer_imgs.value = True
        
            time.sleep(3)
            o = keyframes_list.get_keyframe_by_name('5')
            print('o.slamX, o.slamY, o.slamZ')
            print(o.slamX, o.slamY, o.slamZ)
            print('newer_imgs.value', newer_imgs.value)
            quit()



    while True:
        #print("Inside while")
        # INCREMENTAL RECONSTRUCTION
        #print("INCREMENTAL RECONSTRUCTION")
        #kfrms = os.listdir(cfg.KF_DIR_BATCH / "cam0")
        kfrms = [kf.keyframe_name() for kf in keyframes_list.keyframes()]
        #print(kfrms)
        #print(len(keyframes_list.keyframes()))
        #print(keyframes_list.keyframes()[0]._keyframe_id)
        #print(len(keyframes_list.keyframes_names()))
        #for k in keyframes_list.keyframes_names():
        #    print(type(k))
        #    print(k)
        #print('ok')
        #print()
        #print()
        #kfrms.sort()
        #print('len(kfrms)', len(kfrms))
        #print('newer_imgs', newer_imgs.value)
        if len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION:# and newer_imgs.value == True:
            print("len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs.value == True")
            print("len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs.value == True")
            print("len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs.value == True")
            timer = utils.AverageTimer(logger=logger)
            logger.info(f"DYNAMIC MATCHING WINDOW: {cfg.SEQUENTIAL_OVERLAP}")

            if first_colmap_loop == True:
                logger.info("Initialize an empty database")
                colmap.CreateEmptyDatabase(cfg.DATABASE)
                timer.update("DATABASE INITIALIZATION")

            logger.info("Feature extraction")
            print("Feature extraction")
            if cfg.LOCAL_FEAT_LOCAL_FEATURE != "RootSIFT":
                keypoints, descriptors, laf, kfm_batch = local_feat_extractor.run(cfg.DATABASE, keyframes_list, cfg.IMGS_FROM_SERVER, cfg.KF_DIR_BATCH, cfg.IMG_FORMAT, keypoints, descriptors, laf)
            # Aggiungere LAF per RootSIFT
            elif cfg.LOCAL_FEAT_LOCAL_FEATURE == "RootSIFT":
                colmap.ExtractRootSiftFeatures(
                    database_path=cfg.DATABASE,
                    path_to_images=cfg.KF_DIR_BATCH,
                    first_loop=first_colmap_loop,
                    max_n_features=cfg.LOCAL_FEAT_N_FEATURES,
                )

            cameras.AssignCameras(cfg.DATABASE, len(cfg.CAM))

            print()
            print()
            print()
            print()
            timer.update("FEATURE EXTRACTION")
            print("Sequential matcher")
            logger.info("Sequential matcher")

            if cfg.LOOP_CLOSURE_DETECTION == False:
                if first_colmap_loop == True:
                    old_adjacency_matrix_shape = 0
                elif first_colmap_loop == False:
                    old_adjacency_matrix_shape = adjacency_matrix.shape[0]

                ij = []

                kpoints, des, images = import_local_features.ImportLocalFeature(
                    cfg.DATABASE ######################################################################### Non ha senso importare keypoints, descriptors che poi vengono sovrascritti Fatto per descrittori > 128
                )


                adjacency_matrix = matcher.UpdateAdjacencyMatrix(
                    adjacency_matrix,
                    images.keys(),
                    SEQUENTIAL_OVERLAP,
                    first_colmap_loop,
                    cfg.N_CAMERAS,
                )

                true_indices = np.where(adjacency_matrix)

                #matcher.PlotAdjacencyMatrix(adjacency_matrix)


                if cfg.LOCAL_FEAT_LOCAL_FEATURE == "RootSIFT":
                    keypoints, descriptors = kpoints, des
                    for key in keypoints:
                        laf[key] = None

                #d = new_n_keyframes - old_n_keyframes
                inverted_dict = {value: key for key, value in images.items()}
                print('inverted_dict')
                print(inverted_dict)

                # Matching cam0 at different epochs
                for l, m in zip(true_indices[0], true_indices[1]):
                    if l > m and l > old_adjacency_matrix_shape - 1:
                        kfm1_name = keyframes_list.get_keyframe_by_id(m)._keyframe_name
                        kfm2_name = keyframes_list.get_keyframe_by_id(l)._keyframe_name
                        i = inverted_dict[f"cam0/{kfm2_name}"]
                        j = inverted_dict[f"cam0/{kfm1_name}"]
                        ij.append((i-1, j-1))

                        # Matching between different cameras at the same epoch
                        for c in range(1, cfg.N_CAMERAS):
                            i = inverted_dict[f"cam{c}/{kfm1_name}"]
                            ij.append((i-1, j-1))


                if cfg.LOCAL_FEAT_LOCAL_FEATURE == "LoFTR":
                    matcher.LoFTR(cfg.KF_DIR_BATCH, images, ij, cfg.DATABASE)

                else:
                    for i, j in ij:
                        im1 = images[j + 1]
                        im2 = images[i + 1]

                        if cfg.LOCAL_FEAT_LOCAL_FEATURE == "SuperGlue":
                            kpts1, kpts2, matches2 = matcher.SuperGlue(cfg.KF_DIR_BATCH, im1, im2, local_feat_extractor.detector_and_descriptor.matcher)
                            kpts1 = np.hstack((kpts1, np.zeros((kpts1.shape[0],4))))
                            kpts2 = np.hstack((kpts2, np.zeros((kpts2.shape[0],4))))
                            matches1 = np.arange(len(matches2))
                            mask = matches2 != -1
                            matches1 = matches1[mask]
                            matches2 = matches2[mask]
                            matches_matrix = np.hstack((matches1.reshape(-1,1), matches2.reshape(-1,1)))

                            if matches_matrix.shape[0] < cfg.LOCAL_FEAT_MIN_MATCHES:
                                continue
                            
                            db = db_colmap.COLMAPDatabase.connect(cfg.DATABASE)
                            keypoints = dict(
                                            (image_id, db_colmap.blob_to_array(data, np.float32, (-1, 1)))
                                            for image_id, data in db.execute(
                                                "SELECT image_id, data FROM keypoints"))

                            if int(j + 1) not in keypoints.keys():
                                db.add_keypoints(int(j + 1), kpts1)
                            if int(i + 1) not in keypoints.keys():
                                db.add_keypoints(int(i + 1), kpts2)

                            db.add_two_view_geometry(int(j + 1), int(i + 1), matches_matrix)
                            db.commit()
                            db.close()

                        else:
                            matches_matrix, kps1, kps2 = matcher.Matcher(
                                descriptors[j + 1].astype(float),
                                descriptors[i + 1].astype(float),
                                cfg.KORNIA_MATCHER,
                                cfg.RATIO_THRESHOLD,
                                keypoints[j + 1],
                                keypoints[i + 1],
                                laf[j + 1],
                                laf[i + 1],
                            )

                            if matches_matrix.shape[0] < cfg.LOCAL_FEAT_MIN_MATCHES:
                                continue
                            db = db_colmap.COLMAPDatabase.connect(cfg.DATABASE)
                            db.add_matches(int(j + 1), int(i + 1), matches_matrix)
                            pts1 = keypoints[j + 1][matches_matrix[:, 0], :2].reshape((-1, 2))
                            pts2 = keypoints[i + 1][matches_matrix[:, 1], :2].reshape((-1, 2))
                            if pts1.shape[0] > 8:
                                if cfg.GEOMETRIC_VERIFICATION == "ransac":
                                    F, mask = cv2.findFundamentalMat(
                                        pts1,
                                        pts2,
                                        cv2.FM_RANSAC,
                                        cfg.MAX_ERROR,
                                        cfg.CONFIDENCE,
                                        cfg.ITERATIONS,
                                    )
                                elif cfg.GEOMETRIC_VERIFICATION == "pydegensac":
                                    F, mask = pydegensac.findFundamentalMatrix(
                                        pts1,
                                        pts2,
                                        px_th=cfg.MAX_ERROR,
                                        conf=cfg.CONFIDENCE,
                                        max_iters=cfg.ITERATIONS,
                                        laf_consistensy_coef=-1,
                                        error_type="sampson",
                                        symmetric_error_check=True,
                                        enable_degeneracy_check=True,
                                    )
                                if mask.shape[0] > 8:
                                    try:
                                        if cfg.GEOMETRIC_VERIFICATION == "pydegensac":
                                            mask = mask
                                        elif cfg.GEOMETRIC_VERIFICATION == "ransac":
                                            mask = mask[:, 0]
                                        verified_matches_matrix = matches_matrix[mask, :]
                                        db.add_two_view_geometry(
                                            int(j + 1), int(i + 1), verified_matches_matrix
                                        )
                                    except:
                                        logger.info("No valid geometry")
                                elif mask.shape[0] <= 8:
                                    logger.info("N points < 8")
                            db.commit()
                            db.close()

            # if cfg.LOOP_CLOSURE_DETECTION == False:
            #    colmap.SequentialMatcher(database_path=cfg.DATABASE, loop_closure='0', overlap=str(SEQUENTIAL_OVERLAP), vocab_tree='')

            elif (
                cfg.LOOP_CLOSURE_DETECTION == True
                and cfg.LOCAL_FEAT_LOCAL_FEATURE == "RootSIFT"
            ):
                print("Joining this feature, please contact the author..\nQuit")
                quit()
                colmap.SequentialMatcher(
                    database_path=cfg.DATABASE,
                    loop_closure="1",
                    overlap=str(SEQUENTIAL_OVERLAP),
                    vocab_tree=cfg.VOCAB_TREE,
                )

            else:
                logger.info(
                    "Not compatible option for loop closure detection. Currently only RootSIFT is supported for loop-closure detection. Quit."
                )
                quit()

            timer.update("SEQUENTIAL MATCHER")




            print()
            print()
            print()
            print()
            print('MAPPER')
            logger.info("MAPPER")

            colmap.Mapper(
                database_path=cfg.DATABASE,
                path_to_images=cfg.KF_DIR_BATCH,
                input_path=cfg.OUT_DIR_BATCH,
                output_path=cfg.OUT_DIR_BATCH,
                first_loop=first_colmap_loop,
            )


        if not os.path.exists(f"{cfg.OUT_DIR_BATCH}/0"):
            print("Failed Mapper. Reinitializing..")
            quit() ####################################################################################### da togliere questo quit()
            cfg = init.new_batch_solution()
            first_colmap_loop = True
            for im in processed_imgs:
                os.remove(cfg.CURRENT_DIR / im)

            # Reinit local feature extractor
            local_feat_conf = LocalFeatConfFile(cfg)
            local_feat_extractor = LocalFeatureExtractor(
                cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
            )

            # Setup logging level
            #LOG_LEVEL = logging.INFO
            #utils.Inizialization.setup_logger(LOG_LEVEL)
            #logger = logging.getLogger("ColmapSLAM")

            # Initialize variables
            keyframes_list = KeyFrameList()
            processed_imgs = []
            pointer = 0
            delta = 0
            first_colmap_loop = True
            one_time = False
            reference_imgs = []
            keypoints, descriptors, laf = {}, {}, {}

            # Setup keyframe selector
            kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
            keyframe_selector = KeyFrameSelector(
                keyframes_list=keyframes_list,
                last_keyframe_pointer=pointer,
                last_keyframe_delta=delta,
                keyframes_dir=cfg.KF_DIR_BATCH / "cam0",
                kfs_method=cfg.KFS_METHOD,
                geometric_verification="pydegensac",
                local_feature=cfg.KFS_LOCAL_FEATURE,
                local_feature_cfg=kf_selection_detecor_config,
                n_features=cfg.KFS_N_FEATURES,
                realtime_viz=True,
                viz_res_path=SNAPSHOT_DIR,
                innovation_threshold_pix=cfg.INNOVATION_THRESH_PIX,
                min_matches=cfg.MIN_MATCHES,
                error_threshold=cfg.RANSAC_THRESHOLD,
                iterations=cfg.RANSAC_ITERATIONS,
                n_camera=cfg.N_CAMERAS,

            )

            continue


        timer.update("MAPPER")

        logger.info("Convert model from binary to txt")
        subprocess.run(
            [
                str(cfg.COLMAP_EXE_PATH),
                "model_converter",
                "--input_path",
                cfg.OUT_DIR_BATCH / "0",
                "--output_path",
                cfg.OUT_DIR_BATCH,
                "--output_type",
                "TXT",
            ],
            stdout=subprocess.DEVNULL,
        )

        timer.update("MODEL CONVERSION")
        timer.print("COLMAP")

        # Export cameras
        lines, oriented_dict = export_cameras.ExportCameras(
            cfg.OUT_DIR_BATCH / "images.txt", keyframes_list
        )
        if cfg.DEBUG:
            with open(cfg.OUT_DIR_BATCH / "loc.txt", "w") as file:
                for line in lines:
                    file.write(line)
        print(cfg.DEBUG)

        # Keep track of sucessfully oriented frames in the current kfm_batch
        oriented_kfs_len = 0

        for keyframe in kfm_batch: #for keyframe in keyframes_list.keyframes # for image in kfm_batch
            #if "cam0/" + keyframe.keyframe_name in list(oriented_dict.keys()):
            k = keyframes_list.get_keyframe_by_image_name(Path("imgs/cam0/" + keyframe))
            if "cam0/" + k.keyframe_name() in list(oriented_dict.keys()):
                k.set_oriented()
                oriented_kfs_len += 1

                #for c in range(1, cfg.N_CAMERAS):
                #    if f"cam{c}/" + keyframe.keyframe_name in list(oriented_dict.keys()):
                #        keyframe.slave_cameras[c] = oriented_dict[f"cam{c}/" + keyframe.keyframe_name]
     
        ## Define new reference img (pointer)
        #print(list(oriented_dict.keys()))
        ##print(list(oriented_dict.keys()).glob(f"cam0/*.{cfg.IMG_FORMAT}"))
        #lista = list(oriented_dict.keys())
        ##matching_files = [file for file in lista if glob.fnmatch(file, "cam0/*")]
        ##matching_files = [file for file in lista if any(glob.glob("cam0/*.txt") == file)]
        #cam0_files = [item for item in lista if glob.fnmatch.fnmatch(item, 'cam0/*')]
        #print(cam0_files)
        #quit()
        #last_oriented_keyframe = np.max(list(oriented_dict.keys()))
        #keyframe_obj = keyframes_list.get_keyframe_by_id(last_oriented_keyframe)
        #n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))
        #last_keyframe = keyframes_list.get_keyframe_by_id(n_keyframes - 1)
        #last_keyframe_img_id = last_keyframe.image_id
        #pointer = keyframe_obj.image_id  # pointer to the last oriented image
        #delta = last_keyframe_img_id - pointer
        delta = 0

        # Update dynamic window for sequential matching
        if delta != 0:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP + 2 * (
                n_keyframes - last_oriented_keyframe
            )
            if SEQUENTIAL_OVERLAP > MAX_SEQUENTIAL_OVERLAP:
                SEQUENTIAL_OVERLAP = MAX_SEQUENTIAL_OVERLAP
        else:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP


        ## Report SLAM solution in the reference system of the first image
        
        ref_img_id = list(oriented_dict.keys())[0]
        print('ref_img_id', ref_img_id)
        q0, t0 = oriented_dict[ref_img_id][1]
        t0 = t0.reshape((3, 1))
        q0_quat = Quaternion(q0)
        #print('q0_quat', q0_quat)
        #print('t0', t0)
        #for key in oriented_dict:
        #    cam, keyframe_name = key.split("/", 1)
        #    qi, ti = oriented_dict[key][1]
        #    ti = ti.reshape((3, 1))
        #    qi_quat = Quaternion(qi)
        #    ti_in_q0_ref = (
        #        -np.dot((q0_quat * qi_quat.inverse).rotation_matrix, ti) + t0
        #    )
        #    with lock:
        #        keyframe_obj = keyframes_list.get_keyframe_by_name(keyframe_name)
        #        camera = int(cam[3:])
        #        keyframes_list.add_keyframe(KeyFrame('5','5','5','5','5'))
        #        prova = keyframes_list.get_keyframe_by_name('5')
        #        prova.slamX = 7
        #    
        #        if camera == 0:
        #            print("newer_imgs", newer_imgs)
        #            newer_imgs.value = True
        #            #keyframe_obj.slamX = ti_in_q0_ref[0, 0]
        #            #keyframe_obj.slamY = ti_in_q0_ref[1, 0]
        #            #keyframe_obj.slamZ = ti_in_q0_ref[2, 0]
        #            keyframe_obj.time_last_modification = '0000 aiuto'
        #            keyframe_obj.bug(ti_in_q0_ref[0, 0], ti_in_q0_ref[1, 0], ti_in_q0_ref[2, 0])
        #            
        #            print('cam0', keyframe_obj.image_name())
        #            print(keyframe_obj.slamX, keyframe_obj.slamY, keyframe_obj.slamZ)
        #            print("keyframes_list.debug()")
        #            print("keyframe_obj.time_last_modification", keyframe_obj.time_last_modification)
        #            #keyframes_list.debug()
        #            #quit()
#
#
        #            obj_nuovo = keyframes_list.get_keyframe_by_name(keyframe_name)
        #            print('obj_nuovo', obj_nuovo.slamX, obj_nuovo.slamY, obj_nuovo.slamZ)
        #            quit()
#
#
        #        else:
        #            keyframe_obj.slave_cameras_POS[camera] = (ti_in_q0_ref[0, 0], ti_in_q0_ref[1, 0], ti_in_q0_ref[2, 0])
        #            print('cam1', keyframe_obj.image_name())
        #            print(ti_in_q0_ref[0, 0], ti_in_q0_ref[1, 0], ti_in_q0_ref[2, 0])
        #
        #        print("keyframes_list.debug()")
        #        keyframes_list.debug()
        #        print('speriamo')
        #        keyframe_obj = keyframes_list.get_keyframe_by_name('5')
        #        print(keyframe_obj.slamX, keyframe_obj.slamY, keyframe_obj.slamZ)
        #        print(keyframe_obj.time_last_modification)
        #        print('speriamo')
        #        print('newer_imgs', newer_imgs)
#
#
#
        ##print('DEBUG slamX', keyframes_list.keyframes()[1].slamX)
        #oriented_dict_cam0 = {}
        #for key in oriented_dict:
        #    cam, name = key.split("/", 1)
        #    id, extension = name.split(".", 1)
        #    id = int(id)
        #    if cam == "cam0":
        #        oriented_dict_cam0[id] = oriented_dict[key]
        #oriented_dict = oriented_dict_cam0
#
        #oriented_dict_list = list(oriented_dict.keys())
        #oriented_dict_list.sort()
        #total_kfs_number = len(kfrms)
        #
        ##oriented_kfs_len = len(oriented_dict_list)
        #ori_ratio = oriented_kfs_len / total_kfs_number
#
        #logger.info(
        #    f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}"
        #)
        #print(f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}")
  #
        ## Set scale factor
        #if cfg.N_CAMERAS > 1:
        #    baselines = []
#
        #    #print('DEBUG2')
        #    #for ciao in keyframes_list.keyframes():
        #    #    print('debug',ciao.image_name())
        #    #    print('debug',ciao.keyframe_id())
        #    #    print('debug',ciao.slamX)
        #    #    print('debug',ciao.slamX)
        #    #print("keyframes_list.debug()")
        #    #keyframes_list.debug()
#
        #    for keyframe_id in oriented_dict_list:
        #        print('keyframe_id', keyframe_id)
        #        keyframe_obj = keyframes_list.get_keyframe_by_id(keyframe_id)
        #        print('keyframe_obj.keyframe_name()', keyframe_obj.keyframe_name())
        #        x0 = keyframe_obj.slamX
        #        y0 = keyframe_obj.slamY
        #        z0 = keyframe_obj.slamZ
        #        print('slam', x0, y0, z0)
        #        try:
        #            x1 = keyframe_obj.slave_cameras_POS[1][0]
        #            y1 = keyframe_obj.slave_cameras_POS[1][1]
        #            z1 = keyframe_obj.slave_cameras_POS[1][2]
        #            print('x1, y1, z1', x1, y1, z1)
        #            d = ((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5
        #            baselines.append(d)
        #        except:
        #            pass
        #    
        #    quit() #####################################################################################
        #    #print(baselines)
        #    #print(np.mean(baselines))
        #    scale = cfg.BASELINE_CAM0_CAM1 / np.mean(baselines)
        #    print("mean", np.mean(np.array(baselines)*scale))
        #    print("std", np.std(np.array(baselines)*scale))
        #else:
        #    scale = 1
        #
        ## Apply scale factor
        #with open("./scaled_keyframes1.txt", 'w') as out1, open("./scaled_keyframes2.txt", 'w') as out2:
        #    for keyframe_id in oriented_dict_list:
        #        keyframe_obj = keyframes_list.get_keyframe_by_id(keyframe_id)
        #        keyframe_obj.slamX = keyframe_obj.slamX * scale
        #        keyframe_obj.slamY = keyframe_obj.slamY * scale
        #        keyframe_obj.slamZ = keyframe_obj.slamZ * scale
        #        out1.write(f"{keyframe_id}_0,{keyframe_obj.slamX},{keyframe_obj.slamY},{keyframe_obj.slamZ}\n")
        #        for c in range(1, cfg.N_CAMERAS):
        #            try:
        #                x = keyframe_obj.slave_cameras_POS[c][0] * scale
        #                y = keyframe_obj.slave_cameras_POS[c][1] * scale
        #                z = keyframe_obj.slave_cameras_POS[c][2] * scale
        #                keyframe_obj.slave_cameras_POS[c] = (x, y, z)
        #                out2.write(f"{keyframe_id}_{c},{x},{y},{z}\n")
        #            except:
        #                pass

        # Save keyframes
        with open("./keyframes.pkl", "wb") as f:
            pickle.dump(keyframes_list, f)

        # Report 3D points in ref system of the first image
        with open(f"{cfg.OUT_DIR_BATCH}/points3D.txt", "r") as file:
            lines = file.readlines()

        data = []
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            values = line.split()
            x = float(values[1])
            y = float(values[2])
            z = float(values[3])
            p = np.array([[x], [y], [z]])
            p_trans = np.dot(q0_quat.rotation_matrix, p) + t0
            data.append([p_trans[0, 0], p_trans[1, 0], p_trans[2, 0]])

        with open("./points3D.pkl", "wb") as f:
            pickle.dump(np.array(data), f)

        kfm_batch = []
        first_colmap_loop = False

        ## REINITIALIZE SLAM
        #if (
        #    ori_ratio < cfg.MIN_ORIENTED_RATIO
        #    #or total_kfs_number - oriented_kfs_len > cfg.NOT_ORIENTED_KFMS
        #    or oriented_kfs_len < cfg.NOT_ORIENTED_KFMS
        #):
        #    logger.info(
        #        f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}"
        #    )
        #    logger.info("Not enough oriented images")
#
        #    cfg = init.new_batch_solution()
        #    first_colmap_loop = True
#
        #    for im in processed_imgs:
        #        os.remove(cfg.CURRENT_DIR / im)
#
        #    # Reinit local feature extractor
        #    local_feat_conf = LocalFeatConfFile(cfg)
        #    local_feat_extractor = LocalFeatureExtractor(
        #        cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
        #    )
#
        #    # Setup logging level
        #    #LOG_LEVEL = logging.INFO
        #    #utils.Inizialization.setup_logger(LOG_LEVEL)
        #    #logger = logging.getLogger("ColmapSLAM")
#
        #    # Initialize variables
        #    print("\n\nREINITIALIZE ..")
        #    keyframes_list = KeyFrameList()
        #    processed_imgs = []
        #    pointer = 0
        #    delta = 0
        #    first_colmap_loop = True
        #    one_time = False
        #    reference_imgs = []
        #    keypoints, descriptors, laf = {}, {}, {}
#
        #    # Setup keyframe selector
        #    kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
        #    keyframe_selector = KeyFrameSelector(
        #        keyframes_list=keyframes_list,
        #        last_keyframe_pointer=pointer,
        #        last_keyframe_delta=delta,
        #        keyframes_dir=cfg.KF_DIR_BATCH / "cam0",
        #        kfs_method=cfg.KFS_METHOD,
        #        geometric_verification="pydegensac",
        #        local_feature=cfg.KFS_LOCAL_FEATURE,
        #        local_feature_cfg=kf_selection_detecor_config,
        #        n_features=cfg.KFS_N_FEATURES,
        #        realtime_viz=True,
        #        viz_res_path=SNAPSHOT_DIR,
        #        innovation_threshold_pix=cfg.INNOVATION_THRESH_PIX,
        #        min_matches=cfg.MIN_MATCHES,
        #        error_threshold=cfg.RANSAC_THRESHOLD,
        #        iterations=cfg.RANSAC_ITERATIONS,
        #        n_camera=cfg.N_CAMERAS,
        #    )
#
        #    continue

    timer_global.update(f"{len(kfrms)}")
    time.sleep(cfg.SLEEP_TIME)



if __name__ == '__main__':

    class NewerImage:
         def __init__(self):
             self.value = True

    newer_imgs = NewerImage()
    KFRMS_DIR = Path(r"C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\COLMAP_SLAM\colmap_imgs\0")

    ### SETUP LOGGER
    # Setup logging level. Options are [INFO, WARNING, ERROR, CRITICAL]
    LOG_LEVEL = logging.INFO 
    utils.Inizialization.setup_logger(LOG_LEVEL)
    logger = logging.getLogger()
    logger.info('Setup logger finished')


    ### INITIALIZATION
    MAX_SEQUENTIAL_OVERLAP = 50
    processed_imgs = []
    kfm_batch = []
    pointer = 0  # pointer points to the last oriented image
    delta = 0  # delta is equal to the number of processed but not oriented imgs
    first_colmap_loop = True
    one_time = False  # It becomes true after the first batch of images is oriented
    reference_imgs = []
    adjacency_matrix = None
    keypoints, descriptors, laf = {}, {}, {}

    # Create keyframes
    keyframes_list = KeyFrameList()
    kfrms = os.listdir(KFRMS_DIR / 'cam0')
    print(kfrms)
    kfrms.sort()
    for k in kfrms:
        keyframes_list.add_keyframe(
            KeyFrame(
            'placeholder',
            int(k[:-4]),
            k,
            0,
            int(k[:-4]) + 1)
        )

        print(
                'placeholder',
                int(k[:-4]),
                k,
                0,
                int(k[:-4]) + 1,
            )

    CFG_FILE = "config.ini"
    init = utils.Inizialization(CFG_FILE)
    cfg = init.parse_config_file()
    cfg.DATABASE = Path(r"C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\COLMAP_SLAM\outs\0\db.db")
    cfg.COLMAP_EXE_PATH = Path(r"C:\Users\lmorelli\Desktop\COLMAP\COLMAP-3.6-windows-cuda\COLMAP.bat")
    cfg.KF_DIR_BATCH = Path(r"C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\COLMAP_SLAM\colmap_imgs\0")
    cfg.OUT_DIR_BATCH = Path(r"C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\COLMAP_SLAM\outs\0")

    lock = 'placeholder'
    SEQUENTIAL_OVERLAP = 1
    SNAPSHOT_DIR = 'placeholder'
    processed_imgs = 'placeholder'
    init = 'placeholder'

    ### RUNNING TIME STATISTICS
    cProfile.run('''MappingProcess(
        keyframes_list, 
        logger, 
        cfg, 
        newer_imgs, 
        first_colmap_loop, 
        lock, 
        SEQUENTIAL_OVERLAP, 
        adjacency_matrix, 
        keypoints, 
        descriptors, 
        laf, 
        init, 
        SNAPSHOT_DIR, 
        processed_imgs,
        )''')