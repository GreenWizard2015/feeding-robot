import os, sys
# fix resolving in colab and eclipse ide
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Utils
Utils.setupEnv()

from CAnchorsDetector import CAnchorsDetector, CStackedDetector
import cv2
import numpy as np
from scripts import common

SOURCE_FILE = common.selectSourceFile(0)
FRAME_ID = 475

detector = CStackedDetector([
  CAnchorsDetector().load(kind='xxx', folder='d:/'),
])

cam = cv2.VideoCapture(SOURCE_FILE)
frameID = 0
while True:
  ret, img = cam.read()
  if not ret: break
  frameID += 1
  if frameID == FRAME_ID: break
  continue
cam.release()

def hm2image(pred):
  points, heatmaps = pred
  heatmaps = np.stack(heatmaps)
  stacked = heatmaps.max(axis=0)
  
  cmap = cv2.applyColorMap((stacked * 255).astype(np.uint8), cv2.COLORMAP_JET)
  combined = cv2.addWeighted(img, 0.5, cmap, 0.5, 0)
  mask = np.where(0.01 < stacked)
  res = img.copy()
  res[mask] = combined[mask]
  
  for name, point in points.items():
    print(name, point)
    if not (point is None):
      prob = point['confidence']
      anchor = (point['x'], point['y'])
      color = (0, 0, 255)
      cv2.putText(res, '%s %.0f' % (name, prob * 100), anchor, cv2.FONT_HERSHEY_COMPLEX, .5, color)
      cv2.circle(res, anchor, 3, color, -1)
  return res

cv2.imshow('noise 64', hm2image(
  detector.combinedDetections(
    img, returnHeatmaps=True, crops='all', mode='prod',
    TTA={'mode': 'noise', 'N': 2, 'prod': 0.05, 'use flip': True, 'penalize': True},
    batchSize=4
  )
))

cv2.imshow('flip p', hm2image(
  detector.combinedDetections(
    img, returnHeatmaps=True, crops='all', mode='prod',
    TTA={'mode': 'flip', 'penalize': True}
  )
))

cv2.imshow('flip', hm2image(
  detector.combinedDetections(
    img, returnHeatmaps=True, crops='all', mode='prod',
    TTA={'mode': 'flip', 'penalize': False}
  )
))

# cv2.imshow('noise', hm2image(
#   detector.combinedDetections(
#     img, returnHeatmaps=True, crops='all', mode='prod',
#     TTA={'mode': 'noise'}
#   )
# ))

cv2.waitKey()
