#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../') # fix resolving in colab and eclipse ide

import Utils
Utils.setupEnv()

from CAnchorsDetector import CAnchorsDetector
from scripts import common
from scripts.common import PROJECT_FOLDER, DATASET_FILE
import cv2
import glob
import json
import os

SOURCE_FILE = common.selectSourceFile()

SHOW_PREDICTIONS = False
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
DATASET = common.loadDataset('d:/samples-e000005.json')
# DATASET = common.loadDataset('d:/mined-samples-x.json')
# DATASET = common.loadDataset('d:/mined-samples-1634493039.json')
# DATASET = common.loadDataset('d:/x.json')
print(Utils.datasetLen(DATASET))
VIDEO_ENTITY = DATASET.get(os.path.basename(SOURCE_FILE), {})

def getAnchors(frameID):
  res = [None for _ in ANCHORS_PROP]
  if str(frameID) not in VIDEO_ENTITY:
    return res
  
  entity = VIDEO_ENTITY[str(frameID)]
  entity = entity[0] if isinstance(entity, list) else entity
  for name, value in entity.items():
    ind = next(i for i, x in enumerate(ANCHORS_PROP) if x[0] == name)
    res[ind] = ((value['x'], value['y']), value.get('confidence', 1.0))
  print(Utils.confidence(entity), frameID, sum([1 for x in res if not(x is None)]))
  print(res[-2])
  return res
##########

frameID = 0
nextFrame = None
QUIT = False 
Anchors = None
activeAnchor = None

def toPointStruct(pt):
  (x, y), conf = pt
  res = {'x': x, 'y': y}
  if conf < 1.0:
    res['confidence'] = conf
  return res

def saveToDataset():
  fid = str(frameID)
  if all((x is None) for x in Anchors):
    if fid in VIDEO_ENTITY:
      del VIDEO_ENTITY[fid]
      print('Deleted frame %d of %s from dataset' % (frameID, SOURCE_FILE))
  else:
    info = {
      ANCHORS_PROP[i][0]: toPointStruct(pt) for i, pt in enumerate(Anchors) if pt
    }
    VIDEO_ENTITY[fid] = info
    print('Added frame %d of %s to dataset. Value: %s' % (frameID, SOURCE_FILE, json.dumps(info)))
    
  with open(DATASET_FILE, 'w') as f:
    json.dump(DATASET, f,  indent=2, check_circular=False, sort_keys=True)
  return

def onClickEvent(event, x, y, flags, param):
  pt = ((x, y), 1.0)
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

FIND_NEXT = False
cam = cv2.VideoCapture(SOURCE_FILE)
while not QUIT:
  ret, frame = cam.read()
  if not ret: break
  
  Anchors = getAnchors(frameID)
  predicted = detector.detect(frame) if SHOW_PREDICTIONS else []
  activeAnchor = 0
  nextFrame = False
  if FIND_NEXT:
    if 0 < sum([1 for x in Anchors if not(x is None)]):
      FIND_NEXT = False
    else:
      nextFrame = True
      
  while not (nextFrame or QUIT):
    img = frame.copy()
    for i, anchor in enumerate(Anchors):
      if not (anchor is None):
        text, color = ANCHORS_PROP[i]
        if i == activeAnchor:
          color = (0, 215, 255)
        
        pt, confidence = anchor
        cv2.putText(img, '%s|%.2f' % (text, confidence), pt, cv2.FONT_HERSHEY_COMPLEX, 1, color)
        cv2.circle(img, pt, 8, color)
        
    for i, anchor in enumerate(predicted):
      if not (anchor is None):
        color = (255, 255, 255)
        cv2.circle(img, tuple(int(x) for x in pt[::-1]), 8, color)
        
    cv2.imshow(WINDOW_NAME, img)
    
    key = cv2.waitKey(256) & 0xFF
    QUIT = (key == 27) # escape
    nextFrame = (key == ord('n'))
    
    if (ord('1') <= key) and (key <= ord('8')):
      activeAnchor = key - ord('1')
    
    if ord('f') == key:
      nextFrame = FIND_NEXT = True
  #
  frameID += 1
# Release everything if job is finished
cam.release()
cv2.destroyAllWindows()