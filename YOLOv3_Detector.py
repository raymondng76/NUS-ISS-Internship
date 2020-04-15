# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

import os
import cv2
import sys
import time
import argparse
import numpy as np

class YOLOv3_Detector:
    def __init__(self, 
                nnconfig='cfg/yolov3.cfg', 
                weights='weights/yolov3.weights', 
                classes='data/coco.names',
                conf_thresh=0.5,
                score_thresh=0.5,
                iou_thresh=0.5,
                class_filter=2): # 2 for car in coco dataset
        self.config_path = nnconfig
        self.weights_path = weights
        self.labels = open(classes).read().strip().split("\n")
        self.confidence = conf_thresh
        self.score_threshold = score_thresh
        self.IOU_threshold = iou_thresh
        self.class_filter = class_filter
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
    
    def detect(self, img, draw=False):
        '''
        Detection for single frame, method should return the frame and list of all bounding boxes
        '''
        colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        layer_outputs = self.net.forward(ln)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != self.class_filter:
                    continue
                confidence = scores[class_id]
                if confidence > self.confidence:
                    box = detection[:4] * np.array([w,h,w,h])
                    (centerX, centerY, width, height) = box.astype('int')
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    if x < 0 or y < 0:
                        continue
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.score_threshold, self.IOU_threshold)

        outboxes = []
        if len(idxs) > 0:
                for i in idxs.flatten():
                    outboxes.append(boxes[i])

        if draw:
            font_scale = 1
            thickness = 2
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    color = [int(c) for c in colors[class_ids[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=thickness)
        return img, outboxes

# FOR DEBUG
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--image', type=str, default='', help='Image file to detect on')
#     parser.add_argument('-d', '--draw', action='store_true', help='Flag to draw bounding boxes')
#     parser.add_argument('-o', '--output', type=str, default='outimg.png', help='Filename for output image, only if draw is enabled')
#     args = parser.parse_args()
    
#     img = cv2.imread(args.image)
#     yolo = YOLOv3_Detector()
#     outimg, boxes = yolo.detect(img, args.draw)
#     print(boxes)
#     if args.draw:
#         cv2.imwrite(args.output, outimg)