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

from MVP_utils import ProcessConfig
from MVP_utils import CreateDirIfMissing
from MVP_utils import PadFrame
from MVP_utils import DrawBoundingBoxAndIdx
from MVP_utils import SliceDetection
from MVP_utils import DrawVideoNames

class MultiVideoProcessor:
    '''
    Main class to process multi vid/cam with a detector or tracker and then with ReID
    '''
    def __init__(self, config):
        self.config     = config
        self.detector   = Detector_Tracker_Factory(self.config)
        self.config['reid_algorithm'] = self.config['reid_algorithm'] if self.detector != None else 'NoReID'
        self.reid       = ReID_Factory(self.config) 
        self.vid_stats  = self._processVidStats()
        self.smallestVidKey, self.minNumFrames = self._findMinNumFrame()

        if self.config['verbose'] and not self.config['use_camera']:
            print(f'Detector: [{self.detector}]')
            print(f'ReID: [{self.reid}]')
            print(f'Video with smallest frame: [{self.smallestVidKey}]')
            print(f'Minimium frame count: [{self.minNumFrames}]')
            print('\n')

    def _reid_process_factory(self, reid):
        '''
        Factory method to allow customized reid process for different ReID algorithm
        Add customized processor method as required
        '''
        factory = {
            'PersonReID': self._processPersonReid,
            'DeepPersonReID': self._processPersonReid
        }
        return factory[reid]

    def _processPersonReid(self, frames, boxes, boxes_idx):
        '''
        Method to extract features of image slices and run thru ReID process to generate score
        Implementation is for 'PersonReID' and 'DeepPersonReID' algorithm
        '''
        # ********** QCam processing **********
        # QCam frame, boxes and IDs
        qframe = frames[0]
        qboxes = boxes[0]
        qboxes_idx = boxes_idx[0]
        # Slice detection
        # (Dict) qdet_slice : key = box id, value = crop frame of the bounding box
        qdet_slice = SliceDetection(qframe, qboxes, qboxes_idx)
        # Extract features of qframe
        with torch.no_grad():
            qfeatures = self.reid.extract_features(list(qdet_slice.values()))
        # *************************************

        # ********** GCam processing **********
        reid_idx = {}
        for key in frames.keys():
            if key == 0: # key 0 is for QCam
                continue
            # GCam frame, boxes and IDs
            gframe = frames[key]
            gboxes = boxes[key]
            gboxes_idx = boxes_idx[key]
            # Slice detection
            # (Dict) gdet_slice : key = box id, value = crop frame of the bounding box
            gdet_slice = SliceDetection(gframe, gboxes, gboxes_idx)
            # Extract features of qframe
            with torch.no_grad():
                gfeatures = self.reid.extract_features(list(gdet_slice.values()))
            
            # Run Reid to find matching index
            qScore_idx = self.reid.reid(qfeatures, gfeatures)
            # Change to 1 based index
            outscore = {}
            for iKey in qScore_idx.keys():
                outscore[iKey + 1] = qScore_idx[iKey]
            reid_idx[key] = outscore
        # *************************************
        return reid_idx        

    def _processVidStats(self):
        '''
        Method to process stats of vid or cam.
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
        Stack frames, 3 frame across for each rows
        '''
        hor_stacks = []
        ver_stacks = []
        num_frame = len(frames)
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
            # Pad each frame
            for stk in range(len(reminder_stk)):
                reminder_stk[stk] = PadFrame(reminder_stk[stk])
            # From 2nd row onwards, if there are rows lesser than 3 frames, add blank frames
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
        total_proc_start = time.time()
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
            frames, boxes, boxes_idx, reid_idx = {}, {}, {}, {} 

            # ********** DETECTION / TRACKER **********
            # Grab qCam frame
            qGrabbed, qFrame = self.vid_stats['qCam']['cap'].read()
            if not qGrabbed:
                print('Frame grab from qCam failed!')
                loop = False
                break

            # Run thru detector or tracker
            if self.detector != None:
                if self.config[self.config['det_algorithm']]['type'] == 'detector':
                    outFrame, outBoxes, outBoxesIdx = self.detector.detect(qFrame)
                elif self.config[self.config['det_algorithm']]['type'] == 'tracker':
                    outFrame, outBoxes, outBoxesIdx = self.detector.track(qFrame, 0)
                else: # Just for safety
                    outFrame, outBoxes, outBoxesIdx = qFrame, None, None
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
                    if self.config[self.config['det_algorithm']]['type'] == 'detector':
                        outFrame, outBoxes, outBoxesIdx = self.detector.detect(gFrame)
                    elif self.config[self.config['det_algorithm']]['type'] == 'tracker':
                        outFrame, outBoxes, outBoxesIdx = self.detector.track(gFrame, camidx)
                    else: # Just for safety
                        outFrame, outBoxes, outBoxesIdx = gFrame, None, None
                else:
                    outFrame, outBoxes, outBoxesIdx = gFrame, None, None

                # Assign for gCam, index starts from 1 onwards
                frames[camidx]    = outFrame
                boxes[camidx]     = outBoxes
                boxes_idx[camidx] = outBoxesIdx
            # *****************************************
            
            # ********** REID **********
            if self.reid != None:
                reid_idx = self._reid_process_factory(self.config['reid_algorithm'])(frames, boxes, boxes_idx)                
            # **************************

            # ********** Draw and label bounding boxes **********
            if boxes[0] != None:
                frames = DrawBoundingBoxAndIdx(frames, boxes, boxes_idx, reid_idx)
            # ***************************************************

            # Label frames
            frames = DrawVideoNames(self.config, frames)
            
            # ********** Process and stack frames **********
            frameStack = self._arrangeStack(list(frames.values()))
            # **********************************************

            frameend = time.time()
            frame_time = frameend - framestart
            if self.config['verbose']:
                print(f'FRAME[{frameCounter}] : End Processing -> Processed in {frame_time:.3f} second(s).')

            # ********** Display processed frames **********
            if self.config['display_frames']:
                detname = self.config['det_algorithm']
                reidname = self.config['reid_algorithm']
                title = f'MultiVideo -> Detector_Tracker: [{detname}] -> ReID: [{reidname}]'
                if self.reid != None:
                    title += f' -> ReID match threshold: [{self.reid.threshold}]'
                cv2.imshow(title, frameStack)
            # **********************************************

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

            # Break loop once all frames processed, only for videos
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

        total_proc_end = time.time()
        total_proc_time = (total_proc_end - total_proc_start) / 60
        print(f'Total Processing Time: [{total_proc_time:.3f} minute(s).]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MultiVideoProcessor.py')
    parser.add_argument('-c', '--config', type=str, default='config.ini', help='Config file for all settings')
    args = parser.parse_args()

    # Process config.ini
    cp = configparser.ConfigParser()
    cp.read(args.config)
    config = ProcessConfig(cp)

    if config['verbose']:
        print('\n********** Configuration: **********')
        [print(key, ':', value) for key, value in config.items()]
        print('************************************\n')
    
    # Instantiate MVP and run
    mvp = MultiVideoProcessor(config)
    mvp.run()
