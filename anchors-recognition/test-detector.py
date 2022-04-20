import Utils
Utils.setupEnv()

from CAnchorsDetector import CAnchorsDetector, CStackedDetector
import cv2
import numpy as np
from scripts import common

HEATMAP_OVERLAY = True
LABELS = False
CROPS = 'all' # 'full' or 'all'
SAVE_FRAMES = not True
SAVE_FORMAT = 'jpg'

detector = CAnchorsDetector()
# detector.load(kind='focal-coords-1px', folder='weights')
# detector.load(kind='best-limited', folder='weights')

detector = CStackedDetector([
  CAnchorsDetector().load(kind='limited-0-0', folder='d:/'),
  CAnchorsDetector().load(kind='limited-1-0', folder='d:/'),
#   CAnchorsDetector().load(kind='focal', folder='weights')
])

SOURCE_FILE = common.selectSourceFile()
cam = cv2.VideoCapture(SOURCE_FILE)
QUIT = False
frameID = 0
SKIP = False
while not QUIT:
  ret, img = cam.read()
  if not ret: break
  frameID += 1

  if not SKIP:
    hmScore = 0
    points = {}#detector.detect(img)
    comboPoints, heatmaps = detector.combinedDetections(img, returnHeatmaps=True, crops=CROPS, mode='prod')
    if HEATMAP_OVERLAY:
      heatmaps = np.stack(heatmaps)
      stacked = heatmaps.max(axis=0)
      
      hm = stacked
  #     hm = ((hm/hm.max()) * 255).astype(np.uint8)
      
      binHM = np.where(0.05 < hm, 0, 255).astype(np.uint8)
  #     cv2.imshow('mmmm', 255 - binHM)
      contours, _ = cv2.findContours(255 - binHM, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
      noiseArea = float(sum(cv2.contourArea(x) for x in contours))
      noiseScore = (noiseArea / np.prod(binHM.shape)) * len(contours)
      hmScore += noiseScore
      
      cmap = cv2.applyColorMap((stacked * 255).astype(np.uint8), cv2.COLORMAP_JET)
      combined = cv2.addWeighted(img, 0.5, cmap, 0.5, 0)
      mask = np.where(0.1 < stacked)
      img[mask] = combined[mask]
  
    for i, point in enumerate(points.values()):
      if not (point is None):
        prob = point['confidence']
        anchor = (point['x'], point['y'])
        color = (0, 0, int(prob * 255))
        if LABELS:
          cv2.putText(img, '%d %.0f' % (i, prob * 100), anchor, cv2.FONT_HERSHEY_COMPLEX, .5, color)
        cv2.circle(img, anchor, 3, color, -1)
        
    for i, point in enumerate(comboPoints.values()):
      if not (point is None):
        prob = point['confidence']
        anchor = (point['x'], point['y'])
        color = (0, int(prob * 255), 0)
        if not LABELS:
          cv2.putText(img, '%d %.0f' % (i, prob * 100), anchor, cv2.FONT_HERSHEY_COMPLEX, .5, color)
        cv2.circle(img, anchor, 8, color)
    
    cv2.putText(img, 'Score: %.8f' % (hmScore,), (25, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    pass
  
  if SAVE_FRAMES:
    cv2.imwrite('%06d.%s' % (frameID, SAVE_FORMAT), img)
    print('Saved %06d.%s' % (frameID, SAVE_FORMAT))
  else:
    cv2.imshow('WINDOW_NAME', img)
      
    key = cv2.waitKey(0) & 0xFF 
    QUIT = (key == 27) # escape
    SKIP = (key == ord('s'))
  continue
# Release everything if job is finished
cam.release()
cv2.destroyAllWindows()
