#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../') # fix resolving in colab and eclipse ide

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1 * 1024)]
)

from CAnchorsDetector import CAnchorsDetector
from scripts import common
from scripts.common import PROJECT_FOLDER
import cv2
import glob
import json
import os

SOURCE_FILE = common.selectSourceFile()

SHOW_PREDICTIONS = True
detector = CAnchorsDetector(loadWeights=True) if SHOW_PREDICTIONS else None

ANCHORS_PROP = [
  ('UR', (0, 255, 0)),
  ('UY', (0, 255, 0)),
  ('UW', (0, 255, 0)),
  
  ('DB', (0, 0, 255)),
  ('DG', (0, 0, 255)),
  ('DW', (0, 0, 255)),
  
  ('B1', (0, 0, 0)),
  ('B2', (0, 0, 0)),
]

##########
DATASET_FILE = os.path.join(PROJECT_FOLDER, 'dataset.json')

def loadDataset(filename, video):
  res = {}
  if os.path.exists(filename):
    with open(filename) as f:
      res = json.load(f)
      
  if video not in res:
    res[video] = {}
  return res, res[video]

DATASET, VIDEO_ENTITY = loadDataset(DATASET_FILE, os.path.basename(SOURCE_FILE))

def getAnchors(frameID):
  res = [None for _ in ANCHORS_PROP]
  if str(frameID) not in VIDEO_ENTITY:
    return res
  
  entity = VIDEO_ENTITY[str(frameID)]
  for name, value in entity.items():
    ind = next(i for i, x in enumerate(ANCHORS_PROP) if x[0] == name)
    res[ind] = (value['x'], value['y'])
  return res
##########

frameID = 0
nextFrame = None
QUIT = False 
Anchors = None
activeAnchor = None

def saveToDataset():
  fid = str(frameID)
  if all((x is None) for x in Anchors):
    if fid in VIDEO_ENTITY:
      del VIDEO_ENTITY[fid]
      print('Deleted frame %d of %s from dataset' % (frameID, SOURCE_FILE))
  else:
    info = {
      ANCHORS_PROP[i][0]: {'x': pos[0], 'y': pos[1]} for i, pos in enumerate(Anchors) if pos
    }
    VIDEO_ENTITY[fid] = info
    print('Added frame %d of %s to dataset. Value: %s' % (frameID, SOURCE_FILE, json.dumps(info)))
    
  with open(DATASET_FILE, 'w') as f:
    json.dump(DATASET, f,  indent=2, check_circular=False, sort_keys=True)
  return

def onClickEvent(event, x, y, flags, param):
  pt = (x, y)
  if event == cv2.EVENT_LBUTTONDOWN:
    Anchors[activeAnchor] = pt
    saveToDataset()
    return

  if event == cv2.EVENT_RBUTTONDOWN:
    Anchors[activeAnchor] = None
    saveToDataset()
    return

  return

WINDOW_NAME = 'video'
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, onClickEvent)

cam = cv2.VideoCapture(SOURCE_FILE)
while not QUIT:
  ret, frame = cam.read()
  if not ret: break
  
  Anchors = getAnchors(frameID)
  predicted = detector.detect(frame) if SHOW_PREDICTIONS else []
  activeAnchor = 0
  nextFrame = False
  while not (nextFrame or QUIT):
    img = frame.copy()
    for i, anchor in enumerate(Anchors):
      if not (anchor is None):
        text, color = ANCHORS_PROP[i]
        if i == activeAnchor:
          color = (0, 215, 255)
        cv2.putText(img, text, anchor, cv2.FONT_HERSHEY_COMPLEX, 1, color)
        cv2.circle(img, anchor, 8, color)
        
    for i, anchor in enumerate(predicted):
      if not (anchor is None):
        color = (255, 255, 255)
        cv2.circle(img, tuple(int(x) for x in anchor[::-1]), 8, color)
        
    cv2.imshow(WINDOW_NAME, img)
    
    key = cv2.waitKey(256) & 0xFF 
    QUIT = (key == 27) # escape
    nextFrame = (key == ord('n'))
    
    if (ord('1') <= key) and (key <= ord('8')):
      activeAnchor = key - ord('1')
  #
  frameID += 1
# Release everything if job is finished
cam.release()
cv2.destroyAllWindows()