#!/usr/bin/env python2

import cv2
import time
cap = cv2.VideoCapture('output.avi')

def decodeVid():
  # Read until video is completed
  while(True):
    time.sleep(0.04)
    # Capture frame-by-frame
    ret, frame = cap.read()
    t1 = time.time()
    frame_copy = frame.copy()
    imgRec = frame
    print('time = {}'.format(t1))

decodeVid()

