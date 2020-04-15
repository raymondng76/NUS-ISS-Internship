# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

import os
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

def ProcessVidStats(camList):
    vidList = {}
    for idx in range(len(camList)):
        key = 'cap'