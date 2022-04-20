import os, sys
# fix resolving in colab and eclipse ide
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Utils
Utils.setupEnv()

from CAnchorsDetector import CAnchorsDetector, CStackedDetector
import cv2
import numpy as np

TTA = {'mode': 'flip', 'penalize': True}
MAIN_ARGS = {'crops': 'all', 'mode': 'prod', 'TTA': TTA}
SECONDARY_ARGS = {'crops': 'full', 'mode': 'prod', 'TTA': TTA}
BATCH_SIZE = 164
FRAME_RATE = 10

def collectStats(detector):
  res = []
  for videoSrc, frames in Utils.readAllFrames():
    print('Processing "%s"' % (videoSrc))
    
    for _, frameID, img in frames():
      if 0 < (frameID % FRAME_RATE): continue
      mainPoints = detector.combinedDetections(img, **MAIN_ARGS, raw=True, batchSize=BATCH_SIZE)
      secondaryPoints = detector.combinedDetections(img, **SECONDARY_ARGS, raw=True, batchSize=BATCH_SIZE)
      
      ptsA = np.array([pt for pt, _ in mainPoints], np.float32)
      ptsB = np.array([pt for pt, _ in secondaryPoints], np.float32)
      dist = np.linalg.norm(ptsA - ptsB, axis=-1)
      
      res.append({
        'file': videoSrc,
        'frame': frameID,
        'scores': [score for _, score in mainPoints],
        'distances': dist,
        'main points': mainPoints,
        'secondary points': secondaryPoints,
      })
      continue
    continue
  return res

for model_h5 in CAnchorsDetector.models('weights'):
  modelName = os.path.basename(model_h5)[:-3].replace('anchors-detector-', '')
  print(modelName)
  Utils.save(
    '%s.json' % modelName,
    collectStats(CAnchorsDetector().load(file=model_h5))
  )
  continue