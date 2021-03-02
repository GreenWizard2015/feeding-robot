import cv2
import numpy as np
import random

def rotate_bound(image, angle):
  (h, w) = image.shape[:2]
  (cX, cY) = (h // 2, w // 2)
  # grab the rotation matrix (applying the negative of the
  # angle to rotate clockwise), then grab the sine and cosine
  # (i.e., the rotation components of the matrix)
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  # compute the new bounding dimensions of the image
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))

  # adjust the rotation matrix to take into account translation
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY
  return M, (nW, nH)

def getROI(pts):
  xmin = ymin = float('inf')
  xmax = ymax = -float('inf')
  good = False
  for p in pts:
    if (0 <= p[0]):
      good = True
      xmin = min((xmin, p[0]))
      ymin = min((ymin, p[1]))
      
      xmax = max((xmax, p[0]))
      ymax = max((ymax, p[1]))
  
  if not good:
    xmin = ymin = xmax = ymax = 0
  return np.array((xmin, ymin)), np.array((xmax, ymax))

def brightness_augment(img, factor=0.5):
  return cv2.addWeighted(img, 1. - factor, img, random.random(), random.random())

def sampleAugmentations(
  paddingScale, rotateAngle,
  brightnessFactor=None, noiseRate=0,
  showAugmented=False,
  resize=None
):
  def apply(img, points):
    rotAngle = (random.random() - .5) * 2. * rotateAngle
    transM, HW = rotate_bound(img, rotAngle)
    
    newPoints = np.array([ x[0] for x in cv2.transform(np.array([ [x] for x in points ]), transM) ])
    pts = np.array(points)
    for i, pt in enumerate(points):
      if 0 <= pt[0]:
        pts[i] = newPoints[i]
    points = pts
    ##
    a, b = getROI(points)
    if (b - a).min() < 16:
      return False
    
    middle = (a + b) / 2
    padding = 1.1 + random.random() * paddingScale
    a = middle + (a - middle) * padding
    b = middle + (b - middle) * padding
    
    maxHW = np.array(HW)#[::-1]
    a = np.clip(a, a_min=0, a_max=maxHW)
    b = np.clip(b, a_min=0, a_max=maxHW)
    
    newSize = b - a
    if newSize.min() < 16: return False
    points = [ np.array(x) - a for x in points ]
    
    cX, cY = -a
    transM = np.matmul(
      np.array([[1, 0, cX],[0, 1, cY], [0, 0, 1]]),
      np.vstack([transM, [0,0,1]])
    ).astype(np.float32)[:2]
    
    if not(resize is None):
      scales = np.divide(resize, newSize)[::-1]
      points = [np.array(x) * scales for x in points]
      transM = np.matmul(
        np.array([[scales[0], 0, 0],[0, scales[1], 0], [0, 0, 1]]),
        np.vstack([transM, [0,0,1]])
      ).astype(np.float32)[:2]
      newSize = np.array(resize)

    img = cv2.warpAffine(img, transM[:2], tuple(newSize.astype(np.uint16)), flags=cv2.INTER_NEAREST)
    
    if not (brightnessFactor is None):
      img = brightness_augment(img, factor=brightnessFactor)
    
    if 0 < noiseRate:
      ptsN = int(noiseRate * img.shape[0] * img.shape[1] * (0.1 + 0.9 * random.random()))
      noiseX = np.random.choice(np.arange(img.shape[0]), size=(ptsN,), replace=True)
      noiseY = np.random.choice(np.arange(img.shape[1]), size=(ptsN,), replace=True)
      img[noiseX, noiseY, :] = 0

    if showAugmented:
      cv2.imshow('augmented', img)
    return img, points
  return apply