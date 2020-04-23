# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

from Detector_Tracker.YOLOv3.YOLOv3_Detector import YOLOv3_Detector
from Detector_Tracker.JDE.JDE_Tracker import JDE_Tracker
from ReID.PersonReID.PersonReID import PersonReid
from ReID.DeepPersonReID.DeepPersonReID import DeepPersonReID

def Detector_Tracker_Factory(config):
    '''
    This factory method instantiate and returns the requested detector or tracker algorithm class object
    To add detector, 
        1) Import detector class
        2) Add additional elif statement
        3) Instantiate detector class object
    '''
    detector = None
    algo = config['det_algorithm']

    if algo == 'YOLOv3':
        detector = YOLOv3_Detector(
            nnconfig     = config[algo]['model_config'],
            weights      = config[algo]['weights'],
            classes      = config[algo]['labels'],
            conf_thresh  = float(config[algo]['confidence_threshold']),
            score_thresh = float(config[algo]['score_threshold']),
            iou_thresh   = float(config[algo]['iou_threshold']),
            class_filter = int(config[algo]['class_filter'])
        )

    elif algo == 'JDE':
        detector = JDE_Tracker(
            network_config = config[algo]['network_config'],
            weights        = config[algo]['weights'],
            iou_threshold  = float(config[algo]['iou_threshold']),
            conf_threshold = float(config[algo]['conf_threshold']),
            nms_threshold  = float(config[algo]['nms_threshold']),
            track_buffer   = int(config[algo]['track_buffer']),
            frame_rate     = int(config[algo]['frame_rate']),
            device         = config['device'],
            verbose        = config['verbose'],
            min_box_area   = int(config[algo]['min_box_area']),
            total_cams     = len(config['gCam']) + 1 # 1 for qCam)
        )

    return detector

def ReID_Factory(config):
    '''
    This factory method instantiate and returns the requested ReID algorithm class object
    To add Reid, 
        1) Import Reid class
        2) Add additional elif statement
        3) Instantiate Reid class object
    '''
    reid = None
    algo = config['reid_algorithm']

    if algo == 'PersonReID':
        reid = PersonReid(
            network_config  = config[algo]['network_config'],
            weights         = config[algo]['weights'],
            threshold = float(config[algo]['match_threshold']),
            device          = config['device'],
            verbose         = config['verbose']
        )

    elif algo == 'DeepPersonReID':
        reid = DeepPersonReID(
            model           = config[algo]['model'],
            weights_path    = config[algo]['weights'],
            threshold = float(config[algo]['match_threshold']),
            device          = config['device'],
            verbose         = config['verbose']
        )

    return reid