#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # fix resolving in colab and eclipse ide

import cv2
import glob
import json
import shutil
from collections import namedtuple, defaultdict
import random
import math
import hashlib
from scripts.common import PROJECT_FOLDER, DATASET_FOLDER

CFolderSettings = namedtuple('CFolderSettings', 'folder ratio')

folders = [
  CFolderSettings(os.path.join(DATASET_FOLDER, 'train'), 0.9),
  CFolderSettings(os.path.join(DATASET_FOLDER, 'validation'), 0.1)
]
# cleanup
for x in folders:
  shutil.rmtree(x.folder, ignore_errors=True)
  os.makedirs(x.folder, exist_ok=True)

ANCHORS = ['UR', 'UY', 'UW', 'DB', 'DG', 'DW', 'B1', 'B2']

##########
def loadDataset(filename):
  if os.path.exists(filename):
    with open(filename) as f:
      return json.load(f)
  return {}

DATASET = loadDataset(os.path.join(PROJECT_FOLDER, 'dataset.json'))

def sampleData(video, frame):
  return DATASET[video][str(frame)]
##########
def partitions(ratios, total):
  indByRatio = list(sorted(enumerate(ratios), key=lambda x: x[1]))
  sizes = [1 for _ in ratios]
  maxSize = total
  for ind, ratio in indByRatio:
    sizes[ind] = sz = max((1, min((maxSize, math.ceil(ratio * total))) ))
    maxSize -= sz
  assert total == sum(sizes)
  
  res = []
  prev = 0
  for sz in sizes:
    res.append((prev, prev + sz))
    prev += sz
  return res

samplesByFolder = defaultdict(list)
videoFiles = set()
for name, entity in DATASET.items():
  videoFiles.add(name)
  
  allSamples = []
  for frame in entity.keys():
    allSamples.append((name, frame))
  ##########
  random.shuffle(allSamples)
  for folder, (a, b) in zip(folders, partitions([x.ratio for x in folders], len(allSamples)) ):
    samplesByFolder[folder] += allSamples[a:b]
##########
for folder, samples in samplesByFolder.items():
  print('%s | %d samples' % (folder.folder, len(samples)))
###############
def folderFor(sample):
  for folder, samples in samplesByFolder.items():
    if sample in samples:
      return folder
  return None

samplesInfoByFolders = {x: {} for x in folders}
for videoSrc in videoFiles:
  fullPath = next((x for x in glob.iglob(os.path.join(PROJECT_FOLDER, '**', videoSrc), recursive=True)), None)
  if fullPath is None: continue
  
  frameID = 0
  cam = cv2.VideoCapture(fullPath)
  while True:
    ret, frame = cam.read()
    if not ret: break
    
    sampleID = (videoSrc, str(frameID))
    folder = folderFor(sampleID)
    if folder:
      data = sampleData(videoSrc, frameID)
      uniqID = hashlib.md5(str(
        { 'props': data, 'frame': frameID, 'file': fullPath}
      ).encode('utf8')).hexdigest()
      
      assert uniqID not in samplesInfoByFolders[folder]
      samplesInfoByFolders[folder][uniqID] = data
      
      cv2.imwrite(
        os.path.join(folder.folder, '%s.bmp' % uniqID),
        frame
      )
    frameID += 1
  # Release everything if job is finished
  cam.release()
############
for folder, samples in samplesInfoByFolders.items():
  with open(os.path.join(folder.folder, 'samples.json'), 'w') as f:
    json.dump(samples, f, indent=2)