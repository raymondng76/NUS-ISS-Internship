import os
import cv2
import sys
import time
import datetime
import argparse
import numpy as np
import torch
from YOLOv3_Detector import *
from PersonReID import *
from DeepPersonReID import *

class MultiVideo:
    def __init__(self, args):
        self.args = args
        self.score_threshold = args.score
        self.verbose = args.verbose
        
        self.vidlist = self._getVidPaths(args)
        self.capStats = self._readCapStats()
        self.smallestVidKey, self.minNumFrames = self._findMinNumFrame()
        self.yolov3 = YOLOv3_Detector(tiny_yolo=args.tiny_yolo)

        if not args.disable_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        if args.reid == 'PersonReID':
            self.reid = PersonReid(
                device = device, verbose=self.verbose)
        elif args.reid == 'DeepPersonReID':
            self.reid = DeepPersonReID(
                weights_path=os.path.join('model','osnet_ain_x1_0_mars_softmax_cosinelr','model.pth.tar-150'), 
                device=device, verbose=self.verbose)
        else:
            self.reid = None
            raise ValueError('Requested REID algorithm does not exists!!!')

        self.reid_process_factory = {
                    'PersonReID': self.processPersonReid,
                    'DeepPersonReID': self.processPersonReid
                }
        
    def _getVidPaths(self, args):
        vidpath = []
        file_list = os.listdir(args.videos_path)
        for idx in range(len(file_list)):
            vidpath.append(os.path.join(args.videos_path, file_list[idx]))
        return vidpath

    def _readCapStats(self):
        caplist = {}
        for idx in range(len(self.vidlist)):
            key = 'cap' + str(idx + 1)
            if key not in caplist.keys():
                caplist[key] = {}
            vidcap = cv2.VideoCapture(self.vidlist[idx])
            caplist[key]['cap'] = vidcap
            caplist[key]['n_frames'] = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            caplist[key]['fps'] = int(vidcap.get(cv2.CAP_PROP_FPS))
            caplist[key]['width'] = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            caplist[key]['height'] = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return caplist
    
    def _findMinNumFrame(self):
        assert self.capStats != None
        frameCount = 1e5
        vidIdx = ''
        for idx in range(len(self.capStats)):
            key = 'cap' + str(idx + 1)
            n_frame = self.capStats[key]['n_frames']
            if n_frame < frameCount:
                frameCount = n_frame
                vidIdx = key
        return vidIdx, frameCount
    
    def _arrangeStack(self, frames):
        hor_stacks = []
        ver_stacks = []
        num_frame = len(frames)
        num_complete_stk = int(num_frame/3)
        reminder = num_frame % 3

        # Whole stack
        for idx in range(num_complete_stk):
            sidx = idx * 3
            hstk = np.hstack((frames[sidx], frames[sidx+1], frames[sidx+2]))
            hor_stacks.append(hstk)

        # Reminder stack
        if reminder != 0:
            empty_count = 3 - reminder
            reminder_stk = frames[-reminder:]
            if num_complete_stk != 0:
                empty_frame = (np.zeros(frames[0].shape)).astype('uint8')
                for c in range(empty_count):
                    reminder_stk.append(empty_frame) 
            hor_stacks.append(np.hstack(tuple(reminder_stk)))
        return np.vstack(hor_stacks)
    
    def _slice_detections(self, frame, frame_boxes):
        det_slice = {}
        for bidx in range(len(frame_boxes)):
            x, y = frame_boxes[bidx][0], frame_boxes[bidx][1]
            w, h = frame_boxes[bidx][2], frame_boxes[bidx][3]
            det_slice[bidx] = frame[y:y+h, x:x+w]
        return det_slice

    def _draw_bb_with_id(self, frame, frame_boxes):
        font_scale = 0.8
        thickness = 2
        for bidx in range(len(frame_boxes)):
            x, y = frame_boxes[bidx][0], frame_boxes[bidx][1]
            w, h = frame_boxes[bidx][2], frame_boxes[bidx][3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=thickness)
            text = str(bidx)
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = frame.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(0, 0, 255), thickness=cv2.FILLED)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        return frame
    
    def _draw_bb_with_matched_id(self, frame, frame_boxes, matched_dict):
        font_scale = 0.8
        thickness = 2
        uniq_idx = set(matched_dict.values())
        for fidx in range(len(frame_boxes)):
            x, y = frame_boxes[fidx][0], frame_boxes[fidx][1]
            w, h = frame_boxes[fidx][2], frame_boxes[fidx][3]
            
            gidx_text = str(fidx)
            gtext_offset_x = x
            gtext_offset_y = y + h
            cv2.putText(frame, gidx_text, (int(gtext_offset_x), int(gtext_offset_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=font_scale, color=(0,255,255), thickness=thickness)
            if fidx not in uniq_idx:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=thickness)

        for idx in uniq_idx:
            # indices = [i for i , x in enumerate(matched_dict.values()) if x == idx]
            indices = [k for k in matched_dict.keys() if matched_dict[k] == idx]
            x, y = frame_boxes[idx][0], frame_boxes[idx][1]
            w, h = frame_boxes[idx][2], frame_boxes[idx][3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=thickness)
            text = str(indices).replace('[','').replace(']','')
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = frame.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(255, 0, 255), thickness=cv2.FILLED)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            print(f'Query:{indices}, Gallery: [{idx}]')
        return frame

    def processPersonReid(self, frames, frame_boxes):
        qframe = frames[0]
        qboxes = frame_boxes[0]
        qdet_slice = self._slice_detections(qframe, qboxes)
        frames[0] = self._draw_bb_with_id(qframe, qboxes)
        with torch.no_grad():
            qfeatures = self.reid.extract_features(list(qdet_slice.values()))
        for key in frames.keys():
            if key == 0: # Assume frames from all other videos is the gallery frames
                continue
            gframe = frames[key]
            gboxes = frame_boxes[key]
            gdet_slice = self._slice_detections(gframe, gboxes)
            # Call REID here
            with torch.no_grad():
                gfeatures = self.reid.extract_features(list(gdet_slice.values()))
            qScore_idx = self.reid.reid(qfeatures, gfeatures, self.score_threshold)
            print(f'Process video[{key+1}]')
            frames[key] = self._draw_bb_with_matched_id(gframe, gboxes, qScore_idx)
        return frames

    def showMultiImages(self, detect=False, track=False):
        assert self.capStats != None
        assert self.smallestVidKey

        if self.args.save_vid:
            writer = None
            outpath = self.args.output_path
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
            outpath = os.path.join(outpath, 'output.mp4')
        
        for fidx in range(self.minNumFrames):
            framestart = time.time()
            print(f'FRAME[{fidx+1}]')
            # Grab frames from all videos
            frames = {}
            frame_boxes = {}
            for cidx in range(len(self.capStats)):
                key = 'cap' + str(cidx + 1)
                grabbed, frame = self.capStats[key]['cap'].read()
                if not grabbed:
                    print(f'Read frame from {key} failed!')
                    break
                # Do YOLOv3 Inference
                if detect and not track:
                    frame, _ = self.yolov3.detect(frame, True)
                elif not detect and track:
                    frame, det = self.yolov3.detect(frame, False)
                    frame_boxes[cidx] = det
                # else:
                #     largest_dict = self._getLargestBbox()

                frames[cidx] = frame
            
            # Perform REID here
            # Assume frame from 1st video is the query frame and frame from all other videos are gallery frame
            if track: 
                frames = self.reid_process_factory[self.args.reid](frames, frame_boxes)

            # Stack frames
            vert_stack = self._arrangeStack(list(frames.values()))

            # Display multi frame
            frameend = time.time()
            frame_proc = frameend - framestart
            print(f'Frame[{fidx+1}] processed in {frame_proc:.3f} second(s).\n')
            cv2.imshow(f'Multi-Video -> {str(self.args.reid)} -> Match Score Threshold = {str(self.score_threshold)}', vert_stack)
            # Save videos
            if self.args.save_vid:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'X264')
                    # TODO: remove hardcoded keys
                    writer = cv2.VideoWriter(outpath, 
                                            fourcc, 
                                            self.capStats['cap1']['fps'],
                                            (vert_stack.shape[1], vert_stack.shape[0]),
                                            True)
                writer.write(vert_stack)

            # Press 'q' to stop
            key = cv2.waitKey(args.wait_delay) & 0xFF
            if key == ord('q'):
                break
        
        # Clean up
        if self.args.save_vid:
            writer.release()
        for idx in range(len(self.capStats)):
            key = 'cap' + str(idx + 1)
            self.capStats[key]['cap'].release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MultiVideo.py')
    parser.add_argument('-c', '--config', type=str, default='config.ini', help='Config file for all settings')

    # parser.add_argument('-v', '--videos-path', type=str, default='vid_data\WT', help='Path to videos')
    # parser.add_argument('-w', '--wait-delay', type=int, default=0, help='Delay in ms per frame')
    # parser.add_argument('-s', '--save-vid', action='store_true', help='Save output videos')
    # parser.add_argument('-d', '--detect', action='store_true', help='Draw detection on frame')
    # parser.add_argument('-t', '--track', action='store_true', help='Perform tracking')
    # parser.add_argument('-sc','--score', type=float, default=0.95, help='ReID matching score threshold')
    # parser.add_argument('--reid', type=str, default='PersonReID', help='ReID algorithm to use')
    # parser.add_argument('-ty', '--tiny-yolo', action='store_true', help='Use tiny yolo')
    # parser.add_argument('-vb', '--verbose', action='store_true', help='Verbosity for detailed error message')
    # parser.add_argument('-dc', '--disable-cuda', action='store_true', help='Use CPU instead')
    # parser.add_argument('-o', '--output-path', type=str, default='output', help='Path to save output video')
    args = parser.parse_args()
    args.track = True
    args.detect = False
    args.verbose = True

    

    mv = MultiVideo(args)
    mv.showMultiImages(args.detect, args.track)