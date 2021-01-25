#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../') # fix resolving in colab and eclipse ide

import cv2
import shutil
import os
from datetime import datetime
from scripts.common import PROJECT_FOLDER

destFolder = os.path.join(PROJECT_FOLDER, 'webcam')
os.makedirs(destFolder, exist_ok=True)
cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FOCUS, 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
  os.path.join(destFolder, '%s.avi' % datetime.now().strftime('%m-%d-%y %H-%M-%S')),
  fourcc, 20.0, (1280,  720)
)

while True:
  ret, frame = cam.read()
  if not ret: continue
  
  out.write(frame)
  cv2.imshow('webcam', frame)
  if (cv2.waitKey(1) & 0xFF) == ord('q'): break
  
# Release everything if job is finished
cam.release()
out.release()
cv2.destroyAllWindows()