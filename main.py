# keyframes_list capire se puo' aiutare i tempi non condividere questa variabile fra i processi
# processed_imgs per ora non condivisa tra i processi

import logging
import subprocess
import multiprocessing
import KFrameSelProcess, MappingProcess

from lib import utils
from lib.keyframes import KeyFrameList
from multiprocessing.managers import BaseManager


class CustomManager(BaseManager):
    pass

if __name__ == '__main__':

    LOG_LEVEL = logging.INFO # Options are [INFO, WARNING, ERROR, CRITICAL]
    utils.Inizialization.setup_logger(LOG_LEVEL)
    logger = logging.getLogger()
    logger.info('Setup logger finished')


    logger.info('INITIALIZATION')
    processed_imgs = []
    first_loop = True
    CFG_FILE = "config.ini"
    init = utils.Inizialization(CFG_FILE)
    cfg = init.inizialize()


    logger.info('CAMERA STREAM OR RUN SIMULATOR')
    if cfg.USE_SERVER == True:
        stream_proc = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
    else:
        stream_proc = subprocess.Popen(["python", "./simulator.py"])
    #stream_proc = subprocess.Popen(["python", "./lib/webcam.py"])


    while True:

        if first_loop == True:
            first_loop = False
        elif first_loop == False:
            logger.info('NEW BATCH SOLUTION')
            cfg = init.new_batch_solution()

        logger.info('RUN IN PARALLEL KEYFRAME SELECTION AND MAPPING')
        multiprocessing.freeze_support()
        CustomManager.register('KeyFrameList', KeyFrameList)

        with CustomManager() as Cmanager, multiprocessing.Manager() as manager:
            keyframes_dict = manager.dict()
            keyframes_list = Cmanager.KeyFrameList()
            exit_bool = manager.Value('b', False)
            lock = manager.Lock()
            processed_imgs = manager.list()

            kfrm_process = multiprocessing.Process(
                                                    target=KFrameSelProcess.KFrameSelProcess,
                                                    args=(
                                                        cfg,
                                                        keyframes_list,
                                                        processed_imgs,
                                                        logger,
                                                        lock,
                                                        keyframes_dict,
                                                        exit_bool,
                                                        ))
            
            mapping_process = multiprocessing.Process(target=MappingProcess.MappingProcess, args=(
                                                                                                    keyframes_list,
                                                                                                    logger,
                                                                                                    cfg,
                                                                                                    lock,
                                                                                                    keyframes_dict,
                                                                                                    exit_bool,
                                                                                                    ))
            kfrm_process.start()
            mapping_process.start()
            #kfrm_process.join()
            mapping_process.join()

            logger.info('END')