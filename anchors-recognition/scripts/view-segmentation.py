import os, sys
# fix resolving in colab and eclipse ide
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Utils
Utils.setupEnv()

from CAnchorsDetector import CAnchorsDetector, CStackedDetector
import cv2
import numpy as np
from scripts import common

import matplotlib.colors as COLORS
def replace_with_dict(ar, dic):
  # Extract out keys and values
  k = np.array(list(dic.keys()))
  v = np.array(list(dic.values()))

  # Get argsort indices
  sidx = k.argsort()

  ks = k[sidx]
  vs = v[sidx]
  return vs[np.searchsorted(ks,ar)]

index2color = {
  i: clr for i, clr in enumerate([
    COLORS.ColorConverter.to_rgb(clrX) for clrX in COLORS.CSS4_COLORS.values()
  ])
}
index2color[-1] = (0, 0, 0)

SOURCE_FILE = common.selectSourceFile(0)

detector = CAnchorsDetector(
  networkOpts={'asSegmentation': True, 'segmap': True}
).load(kind='seg', folder='d:/')

cam = cv2.VideoCapture(SOURCE_FILE)
frameID = 0
while True:
  ret, img = cam.read()
  if not ret: break
  imgP = detector.preprocess(img)
  _, _, segmap = detector.network([imgP[None]])
  segmap = segmap.numpy()[0]
  
  segmap[segmap < 8] = -1
  segmap = replace_with_dict(segmap, index2color)
  segmap = cv2.resize(segmap, tuple(img.shape[:2][::-1]))
  cv2.imshow('segmap', segmap)
  
  key = cv2.waitKey(0) & 0xFF 
  QUIT = (key == 27) # escape
  if QUIT: break
  continue
cam.release()

