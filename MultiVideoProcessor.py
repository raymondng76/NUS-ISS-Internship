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
        self.qcam       = self._processVidStats()

    def _processVidStats(self):
        # vidList = {}
        # for idx in range(len(self.))
        pass

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
    mvp.run()
