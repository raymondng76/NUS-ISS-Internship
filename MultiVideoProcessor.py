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
from utils import CreateDirIfMissing
from utils import PadFrame
from utils import DrawBoundingBoxAndIdx
from utils import SliceDetection

class MultiVideoProcessor:
    '''
    Main class to process multi vid/cam with a detector or tracker and then with ReID
    '''
    def __init__(self, config):
        self.config     = config
        self.detector   = Detector_Tracker_Factory(self.config)
        self.reid       = ReID_Factory(self.config)
        self.vid_stats  = self._processVidStats()
        self.smallestVidKey, self.minNumFrames = self._findMinNumFrame()

        if self.config['verbose'] and not self.config['use_camera']:
            print(f'Video with smallest frame: [{self.smallestVidKey}]')
            print(f'Minimium frame count: [{self.minNumFrames}]')

    def _reid_process_factory(self):
        '''
        Method to allow customized reid process for different ReID algorithm
        '''
        factory = {
            'PersonReID': self._processPersonReid,
            'DeepPersonReID': self._processPersonReid
        }

    def _processPersonReid(self):
        pass

    def _processVidStats(self):
        '''
        Private method to process stats of vid or cam.
        '''
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
        '''
        Method to find min num of frame and idx key of vid with smallest amount of frames.
        Used for handling videos with unequal number of frames.
        Not use for cameras.
        '''
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

    def _arrangeStack(self, frames):
        '''
        Stack frames 3 across for each rows
        '''
        # print(f'frames shape: [{frames[0].shape[1], frames[0].shape[0]}]')
        hor_stacks = []
        ver_stacks = []
        num_frame = len(frames)
        print(f'num_frame {num_frame}')
        num_complete_stk = int(num_frame/3)
        reminder = num_frame % 3

        # Whole stack
        for idx in range(num_complete_stk):
            sidx = idx * 3
            hstk = np.hstack((PadFrame(frames[sidx]), PadFrame(frames[sidx+1]), PadFrame(frames[sidx+2])))
            hor_stacks.append(hstk)

        # Reminder stack
        if reminder != 0:
            empty_count = 3 - reminder
            reminder_stk = frames[-reminder:]
            for stk in range(len(reminder_stk)):
                reminder_stk[stk] = PadFrame(reminder_stk[stk])
            
            if num_complete_stk != 0:
                empty_frame = (np.zeros(frames[0].shape)).astype('uint8')
                for c in range(empty_count):
                    reminder_stk.append(PadFrame(empty_frame)) 
            hor_stacks.append(np.hstack(tuple(reminder_stk)))
        return np.vstack(hor_stacks)

    def run(self):
        '''
        Run the demo for multi video/cam
        '''
        assert self.vid_stats != None
        # Handle save video
        if self.config['save_video']:
            writer = None
            CreateDirIfMissing(self.config['save_video_path'])
            outpath = os.path.join(self.config['save_video_path'], self.config['save_video_name'])
        
        # Main loop
        loop = True
        frameCounter = 0
        while(loop):
            frameCounter += 1
            framestart = time.time()

            if self.config['verbose']:
                print(f'FRAME[{frameCounter}] : Start Processing')

            # Variable to store all frames, bounding boxes and idx
            frames    = {}
            boxes     = {}
            boxes_idx = {}
            reid_idx = {}

            # ********** DETECTION / TRACKER **********
            # Grab qCam frame
            qGrabbed, qFrame = self.vid_stats['qCam']['cap'].read()
            if not qGrabbed:
                print('Frame grab from qCam failed!')
                loop = False
                break
            # Run thru detector or tracker
            if self.detector != None:
                outFrame, outBoxes, outBoxesIdx = self.detector.detect(qFrame)
            else:
                outFrame, outBoxes, outBoxesIdx = qFrame, None, None
            # Assign to index 0 for qCam
            frames[0]    = outFrame
            boxes[0]     = outBoxes
            boxes_idx[0] = outBoxesIdx

            # Grab all gCam frame
            for fidx in range(len(self.vid_stats) - 1):
                camidx = fidx + 1
                key = 'gCam' + str(camidx)
                gGrabbed, gFrame = self.vid_stats[key]['cap'].read()
                if not gGrabbed:
                    print(f'Frame grab from {key} failed!')
                    loop = False
                    break
                # Run thru detector or tracker
                if self.detector != None:
                    outFrame, outBoxes, outBoxesIdx = self.detector.detect(gFrame)
                else:
                    outFrame, outBoxes, outBoxesIdx = gFrame, None, None
                # Assign for gCam, index starts from 1 onwards
                frames[camidx]    = outFrame
                boxes[camidx]     = outBoxes
                boxes_idx[camidx] = outBoxesIdx
            # *****************************************
            
            # ********** REID **********
            if self.reid != None:
                pass
            # **************************

            # ********** Draw and label bounding boxes **********
            if boxes[0] != None:
                frames = DrawBoundingBoxAndIdx(frames, boxes, boxes_idx, reid_idx)
            # ***************************************************

            # ********** Process and stack frames **********
            frameStack = self._arrangeStack(list(frames.values()))
            # **********************************************

            frameend = time.time()
            frame_time = frameend - framestart
            if self.config['verbose']:
                print(f'FRAME[{frameCounter}] : End Processing -> Processed in {frame_time:.3f} second(s).\n')

            # ********** Display processed frames **********
            if self.config['display_frames']:
                detname = self.config['det_algorithm']
                reidname = self.config['reid_algorithm']
                cv2.imshow(f'MultiVideo -> Detector_Tracker: [{detname}] -> ReID: [{reidname}]', frameStack)

                # ********** Save video **********
                if self.config['save_video']:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(
                            outpath,
                            fourcc,
                            self.vid_stats['qCam']['fps'],
                            (frameStack.shape[1], frameStack.shape[0]),
                            True
                        )
                    writer.write(frameStack)
                # ********************************

                # Press 'q' to stop
                key = cv2.waitKey(self.config['wait_delay']) & 0xFF
                if key == ord('q'):
                    break
            # **********************************************

            if self.minNumFrames != None:
                if frameCounter >= self.minNumFrames:
                    loop = False

        # ********** Clean up for exit **********
        if self.config['save_video']:
            writer.release()
        self.vid_stats['qCam']['cap'].release()
        for idx in range(len(self.vid_stats) - 1):
            key = 'gCam' + str(idx + 1)
            self.vid_stats[key]['cap'].release()
        cv2.destroyAllWindows()
        # ***************************************

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
    print('\n')
    mvp.run()
