# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

import os
import cv2
import torch

def ConfigSectionMap(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def ListVideoPaths(folder):
    vidpath = []
    file_list = os.listdir(folder)
    for idx in range(len(file_list)):
        vidpath.append(os.path.join(folder, file_list[idx]))
    return vidpath

def ProcessConfig(config):
    out_config = {}
    gconfig = ConfigSectionMap(config, 'General')
    
    # Sort out camera or video files config
    if gconfig['use_camera'] == 'True':
        out_config['use_camera'] = True
        out_config['qCam'] = int(gconfig['query_cam'])
        out_config['gCam'] = list(map(int, gconfig['gallery_cams'].split(',')))
        out_config['wait_delay'] = 1
    else:
        out_config['use_camera'] = False
        out_config['qCam'] = gconfig['query_video']
        out_config['gCam'] = ListVideoPaths(gconfig['gallery_video_folder'])
        out_config['wait_delay'] = int(gconfig['wait_delay'])
    
    # Handle CUDA or CPU
    if gconfig['disable_cuda'] != 'True' and torch.cuda.is_available():
        out_config['device'] = 'cuda'
    else:
        out_config['device'] = 'cpu'

    out_config['save_video'] = True if gconfig['save_video'] == 'True' else False
    out_config['save_video_name'] = gconfig['save_video_name']
    out_config['save_video_path'] = gconfig['save_video_path']
    out_config['verbose'] = True if gconfig['verbose'] == 'True' else False
    out_config['display_frames'] = True if gconfig['display_frames'] == 'True' else False

    # Sort out detector and reid config
    det = gconfig['det_algorithm']
    reid = gconfig['reid_algorithm']
    out_config['det_algorithm'] = det
    out_config['reid_algorithm'] = reid

    det_config = ConfigSectionMap(config, det)
    reid_config = ConfigSectionMap(config, reid)
    
    out_config[det] = {}
    out_config[reid] = {}
    for dkey in det_config.keys():
        out_config[det][dkey] = det_config[dkey]
    for rkey in reid_config.keys():
        out_config[reid][rkey] = reid_config[rkey]

    return out_config

def CreateDirIfMissing(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    
def PadFrame(frame):
    borderType = cv2.BORDER_CONSTANT
    
    top = int(0.0025 * frame.shape[0])
    bottom = top
    left = int(0.0025 * frame.shape[1])
    right = left

    val = [0, 0, 0]

    return cv2.copyMakeBorder(frame, top, bottom, left, right, borderType, None, val)

def DrawBoundingBoxAndIdx(frames, boxes, boxes_idx, reid_idx):
    '''
    frames = Dictionary containing single frame for each cam / vid (frames[0] for qCam, frames[1:] for all gCam)
    boxes = Dictionary containing all bounding boxes for one frame (boxes[0] for qCam, boxes[1:] for all gCam)
    boxes_idx = Dictionary containing all ids of bounding boxes for one frame (boxes_idx[0] for qCam, boxes_idx[1:] for all gCam)
    reid_idx = Dictionary containing all ReID matched ids for all bounding boxes (reid[0] = None, reid[1:] for all gCam)
    '''

    # Draw boxes for QCam
    frames[0] = DrawBBoxforQCam(frames[0], boxes[0], boxes_idx[0])
    
    # Draw boxes for all GCam

    return frames

def DrawBBoxforQCam(frame, boxes, boxes_idx):
    font_scale = 0.8
    thickness  = 2

    # Loop thru all boxes for this frame
    for idx in range(len(boxes)):
        x, y = boxes[idx][0], boxes[idx][1]
        w, h = boxes[idx][2], boxes[idx][3]
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=thickness)
        # Draw bounding box ID text
        id_txt = str(boxes_idx[idx])
        (txt_width, txt_height) = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        txt_offset_x = x
        txt_offset_y = y - 5
        box_coords = (
            (txt_offset_x, txt_offset_y), 
            (txt_offset_x + txt_width + 2, txt_offset_y - txt_height))
        overlay = frame.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(0, 0, 255), thickness=cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, id_txt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=font_scale, thickness=thickness)
    return frame

def SliceDetection(frame, frame_boxes):
    det_slice = {}
    for bidx in range(len(frame_boxes)):
        x, y = frame_boxes[bidx][0], frame_boxes[bidx][1]
        w, h = frame_boxes[bidx][2], frame_boxes[bidx][3]
        det_slice[bidx] = frame[y:y + h, x:x + w]
    return det_slice