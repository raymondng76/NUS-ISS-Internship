# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

import os
import cv2
import sys
import time
import datetime
import argparse
import configparser
import numpy as np
import torch

from Algorithm_Factory import Detector_Tracker_Factory
from Algorithm_Factory import ReID_Factory
from utils import ProcessConfig

class MultiVideoProcessor:
    def __init__(self, config):
        self.config     = config
        self.detector   = Detector_Tracker_Factory(self.config)
        self.reid       = ReID_Factory(self.config)
        self.vid_stats  = self._processVidStats()
        self.smallestVidKey, self.minNumFrames = self._findMinNumFrame()

    def _processVidStats(self):
        vidList = {}
        # Process Query camera / video
        vidList['qCam'] = {}
        qvidcap = cv2.VideoCapture(self.config['qCam'])
        vidList['qCam']['cap'] = qvidcap
        if not self.config['use_camera']:
            vidList['qCam']['n_frames'] = int(qvidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidList['qCam']['fps'] = int(qvidcap.get(cv2.CAP_PROP_FPS))
        vidList['qCam']['width'] = int(qvidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vidList['qCam']['height'] = int(qvidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Process all Gallery camera / video
        gcamList = self.config['gCam']
        for idx in range(len(gcamList)):
            key = 'gCam' + str(idx + 1)
            if key not in vidList.keys():
                vidList[key] = {}
            gvidcap = cv2.VideoCapture(gcamList[idx])
            vidList[key]['cap'] = gvidcap
            if not self.config['use_camera']:
                vidList[key]['n_frames'] = int(gvidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidList[key]['fps'] = int(gvidcap.get(cv2.CAP_PROP_FPS))
            vidList[key]['width'] = int(gvidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vidList[key]['height'] = int(gvidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return vidList

    def _findMinNumFrame(self):
        assert self.vid_stats != None
        if self.config['use_camera']:
            return None, None
        frameCount = self.vid_stats['qCam']['n_frames']
        vidIdx = 'qCam'
        for idx in range(len(self.vid_stats) - 1):
            key = 'gCam' + str(idx + 1)
            n_frame = self.vid_stats[key]['n_frames']
            if n_frame < frameCount:
                frameCount = n_frame
                vidIdx = key
        return vidIdx, frameCount

    def run(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MultiVideoProcessor.py')
    parser.add_argument('-c', '--config', type=str, default='config.ini', help='Config file for all settings')
    args = parser.parse_args()

    cp = configparser.ConfigParser()
    cp.read(args.config)
    config = ProcessConfig(cp)

    if config['verbose']:
        print('Configuration:')
        [print(key, ':', value) for key, value in config.items()]
        
    mvp = MultiVideoProcessor(config)
    print(mvp.vid_stats)
    print(f'smallest vid key: {mvp.smallestVidKey}')
    print(f'min num frame: {mvp.minNumFrames}')
    mvp.run()
