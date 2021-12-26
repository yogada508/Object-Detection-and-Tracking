#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/TownCentreXVID.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
#list = [[] for _ in range(100)]

def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)

    iou_area = iou_w * iou_h
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area

    return max(iou_area/all_area, 0)

def main(yolo):

    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    frame_index = -1
    video_capture = cv2.VideoCapture(2)

    ###########
    # initialize frame for movement detector
    ret, frame = video_capture.read()
    avg = cv2.blur(frame, (4, 4))
    avg_float = np.float32(avg)
    ###########


    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        ######################
        # movement detector
        # 模糊處理
        blur = cv2.blur(frame, (4, 4))
        # 計算目前影格與平均影像的差異值
        diff = cv2.absdiff(avg, blur)
        # 將圖片轉為灰階
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # 篩選出變動程度大於門檻值的區域
        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        # 使用型態轉換函數去除雜訊
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 產生等高線
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moving_boxes = []
        for c in contours:
            # 忽略太小的區域
            if cv2.contourArea(c) < 2500:
                continue
            # 偵測到物體，可以自己加上處理的程式碼在這裡...

            # 計算等高線的外框範圍
            (x, y, w, h) = cv2.boundingRect(c)
            moving_boxes.append((x,y,w,h))
            # 畫出外框
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 畫出等高線（除錯用）
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
        ######################
        

        #image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs, confidence, class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []

        #yolo bounding box
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            #print(class_names)
            #print(class_names[p])

        #deep sort bounding box
        #sort bounding box's order by the x coordinate in each box
        sort_tracks = sorted(tracker.tracks, key = lambda x: x.to_tlbr()[0])
        moving_record = []
        for track in sort_tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            #print(frame_index)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            b0 = bbox[0]#.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = bbox[1]#.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = bbox[2]-bbox[0]#.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = bbox[3]-bbox[1]

            #calculate each person's moving ratio 
            iou_sum = 0
            for j in moving_boxes:
                iou = get_iou(j,(b0,b1,b2,b3))
                iou_sum += iou
            moving_record.append((track.track_id,iou_sum))

            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

			# draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

                
        print(moving_record)

        count = len(set(counter))
        cv2.putText(frame, "Total Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Pedestrian Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO4_Deep_SORT', frame)


        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ##############
        #update frame for movement detector, the last argument is updating rate
        cv2.accumulateWeighted(blur, avg_float, 0.2)
        avg = cv2.convertScaleAbs(avg_float)
        #####################


    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")
	#print("[INFO]: model_image_size = (960, 960)")
    video_capture.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
