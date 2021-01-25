import tensorflow as tf
import os
import glob
from CAnchorsDetector import CAnchorsDetector

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2 * 1024)]
)

from BasicDataGenerator import CBasicDataGenerator
from sampleAugmentations import sampleAugmentations
import cv2
import numpy as np
from scripts import common

SOURCE_FILE = common.selectSourceFile()
detector = CAnchorsDetector()
detector.load(kind='best')

cam = cv2.VideoCapture(SOURCE_FILE)
QUIT = False
while not QUIT:
  ret, img = cam.read()
  if not ret: break
  
  comboPoints = detector.combinedDetections(img)
  points = detector.detect(img, returnProbabilities=True)
  for i, point in enumerate(points):
    if not (point is None):
      anchor, _, areaProb = point
      areaProb = areaProb[.1 < areaProb].reshape(-1)
      prob = areaProb.max() if 0 < areaProb.size else 0 
      anchor = tuple(int(x) for x in anchor[::-1])
      color = (0, 0, int(prob * 255))
      cv2.putText(img, '%d %.0f' % (i, prob * 100), anchor, cv2.FONT_HERSHEY_COMPLEX, .5, color)
      cv2.circle(img, anchor, 8, color)
      
  for i, point in enumerate(comboPoints):
    if not (point is None):
      anchor, prob = point 
      anchor = tuple(int(x) for x in anchor[::-1])
      color = (0, int(prob * 255), 0)
      cv2.putText(img, '%d %.0f' % (i, prob * 100), anchor, cv2.FONT_HERSHEY_COMPLEX, .5, color)
      cv2.circle(img, anchor, 8, color)
      
  cv2.imshow('WINDOW_NAME', img)
    
  key = cv2.waitKey(20) & 0xFF 
  QUIT = (key == 27) # escape
  
# Release everything if job is finished
cam.release()
cv2.destroyAllWindows()