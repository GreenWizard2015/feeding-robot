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
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert to hsv
  hsv = np.array(hsv, dtype=np.float64)
  hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
  hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
  rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
  return rgb

def sampleAugmentations(
  paddingScale, rotateAngle,
  brightnessFactor=None, noiseRate=0,
  showAugmented=False
):
  def apply(img, points):
    if 0 < rotateAngle:
      rotAngle = (random.random() - .5) * 2. * rotateAngle
      M, HW = rotate_bound(img, rotAngle)
      img = cv2.warpAffine(img, M, HW)
      
      newPoints = np.array([ x[0] for x in cv2.transform(np.array([ [x] for x in points ]), M) ])
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
    
    maxHW = np.array(img.shape[:2])[::-1]
    a = np.clip(a, a_min=0, a_max=maxHW)
    b = np.clip(b, a_min=0, a_max=maxHW)
    
    if (b - a).min() < 16: return False
    points = [ np.array(x) - a for x in points ]
    img = img[int(a[1]):int(b[1])+1, int(a[0]):int(b[0])+1]
    
    if not (brightnessFactor is None):
      img = brightness_augment(img, factor=brightnessFactor)
    
    if 0 < noiseRate:
      noise = (1. - noiseRate) < np.random.random_sample(size=img.shape[:2])
      img[noise] *= 0

    if showAugmented:
      cv2.imshow('augmented', img)
    return img, points
  return apply