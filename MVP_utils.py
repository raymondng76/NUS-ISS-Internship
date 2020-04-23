# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

import os
import cv2
import torch

# Constants for drawing bounding boxes and ID
FONT_SCALE           = 0.7
FONT_THICKNESS       = 1
BOX_LINE_THICKNESS   = 2
VNAME_FONT_THICKNESS = 2
# COLOR Constants
YELLOW               = (0, 255, 255)
RED                  = (0, 0, 255)
BLUE                 = (255, 0, 0)
BLACK                = (0, 0, 0)
ORANGE               = (0, 165, 255)

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
    top    = int(0.005 * frame.shape[0])
    bottom = top
    left   = int(0.0025 * frame.shape[1])
    right  = left
    val = [0, 0, 0]
    return cv2.copyMakeBorder(frame, top, bottom, left, right, borderType, None, val)

def DrawVideoNames(config, frames):
    # Draw qCam video name
    qFrame = frames[0]
    qVidName = config['qCam'] if not config['use_camera'] else 'Camera ' + config['qCam']
    # Draw overlay box
    (txt_width, txt_height) = cv2.getTextSize(qVidName, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=FONT_THICKNESS)[0]
    txt_offset_x = 5
    txt_offset_y = 25 - 5
    box_coords = (
        (txt_offset_x, txt_offset_y), 
        (txt_offset_x + txt_width + 2, txt_offset_y - txt_height))
    overlay = qFrame.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=BLACK, thickness=cv2.FILLED)
    # Draw text
    qFrame = cv2.addWeighted(overlay, 0.6, qFrame, 0.4, 0)
    cv2.putText(qFrame, qVidName, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, color=ORANGE, fontScale=FONT_SCALE, thickness=VNAME_FONT_THICKNESS)
    frames[0] = qFrame

    # Draw gCam video name
    for idx in range(len(frames) - 1):
        gidx = idx + 1
        gFrame = frames[gidx]
        gVidName = config['gCam'][idx] if not config['use_camera'] else 'Camera ' + config['gCam'][idx]
        # Draw overlay box
        (txt_width, txt_height) = cv2.getTextSize(gVidName, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=VNAME_FONT_THICKNESS)[0]
        txt_offset_x = 5
        txt_offset_y = 25 - 5
        box_coords = (
            (txt_offset_x, txt_offset_y), 
            (txt_offset_x + txt_width + 2, txt_offset_y - txt_height))
        overlay = gFrame.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=BLACK, thickness=cv2.FILLED)
        # Draw text
        gFrame = cv2.addWeighted(overlay, 0.6, gFrame, 0.4, 0)
        cv2.putText(gFrame, gVidName, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, color=ORANGE, fontScale=FONT_SCALE, thickness=VNAME_FONT_THICKNESS)
        frames[gidx] = gFrame
    return frames

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
    for idx in range(len(frames) - 1):
        gidx = idx + 1
        reid = reid_idx[gidx] if bool(reid_idx) else None
        frames[gidx] = DrawBBoxforGCams(frames[gidx], boxes[gidx], boxes_idx[gidx], reid) 
    return frames

def DrawBBoxforQCam(frame, boxes, boxes_idx):
    '''
    Method to draw bbox and label for query cam
    '''
    # Loop thru all boxes for this frame
    for idx in range(len(boxes)):
        x, y = int(boxes[idx][0]), int(boxes[idx][1])
        w, h = int(boxes[idx][2]), int(boxes[idx][3])
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=RED, thickness=BOX_LINE_THICKNESS)
        # Draw bounding box ID text
        id_txt = str(boxes_idx[idx])
        (txt_width, txt_height) = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=FONT_THICKNESS)[0]
        txt_offset_x = x
        txt_offset_y = y - 5
        box_coords = (
            (txt_offset_x, txt_offset_y), 
            (txt_offset_x + txt_width + 2, txt_offset_y - txt_height))
        overlay = frame.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=RED, thickness=cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, id_txt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, color=BLACK, fontScale=FONT_SCALE, thickness=FONT_THICKNESS)
    return frame

def DrawBBoxforGCams(frame, boxes, boxes_idx, reid_idx):
    '''
    Method to draw bbox and label for gallery cam
    '''
    if reid_idx != None:
        unique_idx = set(reid_idx.values())
    else:
        unique_idx = {}
    # Loop thru all boxes for this frame
    for idx in range(len(boxes)):
        # Detection / Tracked box
        x, y = int(boxes[idx][0]), int(boxes[idx][1])
        w, h = int(boxes[idx][2]), int(boxes[idx][3])
        # ID from detection / track
        detID_txt = str(boxes_idx[idx])
        detID_offset_x = x
        detID_offset_y = y + h # Print detection ID at bottom
        cv2.putText(frame, detID_txt, (int(detID_offset_x), int(detID_offset_y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONT_SCALE, color=YELLOW, thickness=FONT_THICKNESS)
        if idx not in unique_idx:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=YELLOW, thickness=BOX_LINE_THICKNESS)
    
    # Loop thru all unique matched ReID
    for uidx in unique_idx:
        indices = [k for k in reid_idx.keys() if reid_idx[k] == uidx]
        # Matched boxes
        x, y = int(boxes[uidx][0]), int(boxes[uidx][1])
        w, h = int(boxes[uidx][2]), int(boxes[uidx][3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=BLUE, thickness=BOX_LINE_THICKNESS)
        # Matched ID
        text = str(indices).replace('[','').replace(']','')
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=FONT_THICKNESS)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = frame.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(255, 0, 255), thickness=cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONT_SCALE, color=(0, 0, 0), thickness=FONT_THICKNESS)
    return frame

def SliceDetection(frame, boxes, boxes_idx):
    det_slice = {}
    for bidx in range(len(boxes)):
        x, y = int(boxes[bidx][0]), int(boxes[bidx][1])
        w, h = int(boxes[bidx][2]), int(boxes[bidx][3])
        det_slice[boxes_idx[bidx]] = frame[y:y + h, x:x + w]
    return det_slice