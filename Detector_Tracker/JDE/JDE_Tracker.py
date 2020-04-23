# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
#
# Code is adapted from : https://github.com/Zhongdao/Towards-Realtime-MOT
# ------------------------------

from Detector_Tracker.JDE.tracker.multitracker import JDETracker
from Detector_Tracker.JDE.utils.parse_config import parse_model_cfg
import cv2
import torch
import numpy as np

class JDE_Tracker:
    '''
    This is the wrapper class for the JDETracker algorithm model instance
    Due to multi video processing, one instance of the JDETracker is required for each video being processed
    '''
    def __init__(self, 
        network_config, 
        weights, 
        iou_threshold, 
        conf_threshold,
        nms_threshold,
        track_buffer, 
        device, 
        frame_rate, 
        verbose, 
        min_box_area,
        total_cams):

        self.network_config = network_config
        cfg_dict            = parse_model_cfg(self.network_config)
        self.img_size       = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]
        self.iou_threshold  = iou_threshold
        self.conf_threshold = conf_threshold
        self.verbose        = verbose
        self.frame_rate     = frame_rate
        self.min_box_area   = min_box_area

        # Each camera/video need to have its own tracker object as the tracklets needs to persist for tracking purposes
        self.JDETracker = []
        for idx in range(total_cams):
            self.JDETracker.append(JDETracker(
                network_config, 
                weights, 
                iou_threshold, 
                conf_threshold, 
                nms_threshold, 
                self.img_size, 
                track_buffer, 
                device, 
                frame_rate))  

        if self.verbose:
            print(f'********** JDE_Tracker **********')
            print(f'IOU Threshold : [{self.iou_threshold}]')
            print(f'CONF Threshold : [{self.conf_threshold}]')
            print(f'Frame Rate : [{self.frame_rate}]')
            print(f'Img Size : [{self.img_size}]')
            print(f'Min Box Area : [{self.min_box_area}]')
            print(f'*********************************')
    
    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw *a), int(vh*a)

    def processImage(self, frame):
        w, h = self.get_size(frame.shape[1], frame.shape[0], self.img_size[0], self.img_size[1])
        # Resize
        img0 = cv2.resize(frame, (w, h))
        # Padded resize
        img, _, _, _ = self.letterbox(img0, height=self.img_size[1], width=self.img_size[0])
        # Normalize RGB
        img = img[:,:,::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img, img0

    def track(self, frame, cam):
        # Start tracking based on frame input and which cam
        img, img0 = self.processImage(frame)
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = self.JDETracker[cam].update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        return img0, online_tlwhs, online_ids

    def letterbox(self, img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height)/shape[0], float(width)/shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh