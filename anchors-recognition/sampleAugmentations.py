import cv2
import numpy as np
import random

def rotate_bound(imageHW, angle, pointC=None):
  (h, w) = imageHW
  (cX, cY) = (w // 2, h // 2) if pointC is None else pointC

  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))

  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY
  return M, (nW, nH)

def _isVisible(p, wh):
  return (0 <= p[0] < wh[0]) and (0 <= p[1] < wh[1])

def getROI(pts, wh):
  mask = np.all(np.logical_and(0 <= pts, pts < wh), axis=-1, keepdims=True)
  minV = np.min(pts, where=mask, initial=np.inf, axis=0)
  if np.inf == minV[0]:
    return np.array(((0, 0), wh), np.float32)
  
  maxV = np.max(pts, where=mask, initial=0, axis=0)
  return (minV, maxV)

def brightness_augment(img, factor):
  return cv2.addWeighted(img, factor, img, 0, 16 * random.uniform(-1., 1.))

def randomRange(params):
  if isinstance(params, tuple):
    a, b = params
  else:
    a, b = 0, params

  diff = b - a
  def f():
    return a + random.random() * diff
  return f

def apply_motion_blur(image, size, angle):
  k = np.zeros((size, size), dtype=np.float32)
  k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
  k = cv2.warpAffine(
    k,
    cv2.getRotationMatrix2D((size / 2 - 0.5 , size / 2 - 0.5 ), angle, 1.0),
    (size, size)
  )  
  k = k * ( 1.0 / np.sum(k) )        
  return cv2.filter2D(image, -1, k) 

def sampleAugmentations(
  paddingScale, rotateAngle,
  paddingClipping=(-0.1, 1.1),
  brightnessFactor=None,
  noiseRate=0, noiseMode='RGB',
  multiplicativeNoise=0.1,
  showAugmented=False,
  resize=None,
  aspectRatioRange=(0, float('inf')),
  minSize=64,
  minDistance=0.02,
  maxFails=100,
  maxCandidates=5,
  borderMode=cv2.BORDER_REPLICATE,
  flipX=0.5,
  flipY=0.5,
  motionBlurStrength=(0, 10),
  motionBlurAngle=(-180, 180),
  KPMaskRate=0.05,
  KPMaskSizeRange=(3, 20),
):
  KPMaskSizeMin, KPMaskSizeMax = KPMaskSizeRange
  paddingScale = randomRange(paddingScale)
  rotateAngle = randomRange(rotateAngle)
  brightnessFactor = None if brightnessFactor is None else randomRange(brightnessFactor)
  noiseRate = randomRange(noiseRate) if 0 < noiseRate else None
  
  motionBlurStrength = randomRange(motionBlurStrength)
  motionBlurAngle = randomRange(motionBlurAngle)
  
  def applyFlip(img, points):
    H, W, _ = img.shape
    if random.random() < flipX:
      img = cv2.flip(img, 1)
      points = np.array([(W - x, y) for x, y in points])
    ######
    if random.random() < flipY:
      img = cv2.flip(img, 0)
      points = np.array([(x, H - y) for x, y in points])
    ######
    return img, points
  
  def applyNoise(img):
    if 0.0 < multiplicativeNoise:
      noise = 1 + ((np.random.random(img.shape) - 0.5) * multiplicativeNoise * 2)
      img = np.clip(img * noise, 0, 255).astype(img.dtype)

    if not (noiseRate is None):
      ptsN = int(noiseRate() * img.shape[0] * img.shape[1])
      noiseX = np.random.choice(np.arange(img.shape[0]), size=(ptsN,), replace=True)
      noiseY = np.random.choice(np.arange(img.shape[1]), size=(ptsN,), replace=True)
      
      noise = 0
      if 'RGB' == noiseMode:
        noise = np.random.random(size=(ptsN, 3)) * 255
        
      img[noiseX, noiseY, :] = noise
    return img

  def applyKPMasks(img, points):
    for i, pt in enumerate(points):
      if KPMaskRate <= random.random(): continue
      x, y = pt
      if (
        ((x < 0.0) or (img.shape[0] <= int(x))) or 
        ((y < 0.0) or (img.shape[1] <= int(y)))
      ): continue
      
      D = min([
        np.max(np.abs(pt - ptB)) for ptB in points if not np.allclose(pt, ptB)
      ]) / 2.0
      D = min((D, x, y, img.shape[0] - x, img.shape[1] - y))
      if D <= KPMaskSizeMin: continue
      D = int(min((D, KPMaskSizeMax)))
      
      for _ in range(100):
        mx = random.randint(D, img.shape[0] - 1 - D)
        my = random.randint(D, img.shape[1] - 1 - D)
        dist = min([
          np.abs(np.subtract((mx, my), ptB)).max() for ptB in points
        ])
        dist = min((dist, mx, my, img.shape[0] - mx, img.shape[1] - my))
        if dist < D: continue
        maskArea = img[my-D:my+D+1, mx-D:mx+D+1]
        img[int(y)-D:int(y)+D+1, int(x)-D:int(x)+D+1] = maskArea
        points[i, :] = -img.shape[0]
        break        
      continue
    return img, points
    
  def affineTransformations(img, srcPoints, pointsMin, validPointsMask, N, includeZeroPoints):
    imgHW = img.shape[:2]
    res = []
    resCombinations = set()
    srcPoints = np.array([[*x, 1] for x in srcPoints], np.float32).T
    maskedPoints = ~validPointsMask
    for _ in range(maxFails):
      transM, WH = rotate_bound(imgHW, rotateAngle())
      points = np.dot(transM, srcPoints).T
      points[maskedPoints] = -10 * max(WH)
      ##
      a, b = getROI(points, WH)
      middle = (a + b) / 2
      delta = (middle - a) * paddingScale()
      
      pcMin, pcMax = paddingClipping
      pcMin = np.multiply(WH, pcMin)
      pcMax = np.multiply(WH, pcMax)
      a = np.clip(middle - delta, a_min=pcMin, a_max=pcMax)
      b = np.clip(middle + delta, a_min=pcMin, a_max=pcMax)

      startPt = a + np.random.random((2, )) * (b - a)
      endPt = startPt + np.random.random((2, )) * (b - startPt)
      newSize = endPt - startPt
      if newSize.min() < minSize: continue
      
      aspectRatio = newSize[0] / (1 + newSize[1])
      if not (aspectRatioRange[0] <= aspectRatio <= aspectRatioRange[1]): continue
      points -= startPt
      
      (visibleInd,) = np.where(
        np.all(np.logical_and(0 <= points, points < newSize), axis=-1)
      )
      # naive impl. :(
      KPDist = float('inf')
      for a in points[visibleInd]:
        for b in points[visibleInd]:
          d = np.linalg.norm((a - b) / newSize)
          if 0 < d < KPDist:
            KPDist = d
          continue
        continue
      
      if KPDist < minDistance: continue
      
      visibleInd = tuple(visibleInd)
      if visibleInd in resCombinations: continue
      resCombinations.add(visibleInd)
      if len(visibleInd) < pointsMin:
        if not includeZeroPoints: continue
        includeZeroPoints = False
      
      cX, cY = -startPt
      transM = np.matmul(
        np.array([[1, 0, cX],[0, 1, cY], [0, 0, 1]]),
        np.vstack([transM, [0,0,1]])
      ).astype(np.float32)[:2]
      
      res.append((points, newSize, transM, visibleInd))
      if N <= len(res): break
      continue
    return res
  
  def EMTransformations(img, srcPoints, validPointsMask, W, N):
    imgHW = img.shape[:2]
    P = np.cumsum(W.reshape(-1))
    fract = np.divide(imgHW, W.shape[:2])
    
    res = []
    srcPoints = np.array([[*x, 1] for x in srcPoints], np.float32).T
    maskedPoints = ~validPointsMask
    for _ in range(maxFails):
      sampledPt = np.array(np.unravel_index(
        np.searchsorted(P, random.random() * P[-1]),
        W.shape[:2]
      ), np.float32)
      sampledPt = (0.5 + sampledPt) * fract
      sampledPt += np.random.uniform(low=-fract/2.0, high=fract/2.0, size=sampledPt.shape)
      
      transM, WH = rotate_bound(imgHW, rotateAngle(), pointC=sampledPt)
      ##
      WH = np.array(WH, np.float32)
      middle = WH / 2.0
      pcMin, pcMax = paddingClipping
      pcMin = np.multiply(WH, pcMin)
      pcMax = np.multiply(WH, pcMax)
      maxHW = pcMax - pcMin
      
      ARatio = random.uniform(aspectRatioRange[0], aspectRatioRange[1])
      nW = random.uniform(minSize, min((maxHW[0], maxHW[1] / ARatio)))
      newSize = np.array((nW, nW * ARatio))
      
      delta = newSize / 2.0
      startPt = np.clip(middle - delta, a_min=pcMin, a_max=pcMax)
      endPt = np.clip(middle + delta, a_min=pcMin, a_max=pcMax)
      newSize = endPt - startPt
      if newSize.min() < minSize: continue
      
      points = np.dot(transM, srcPoints).T
      points[maskedPoints] = -10 * max(WH)
      points -= startPt
      
      (visibleInd,) = np.where(
        np.all(np.logical_and(0 <= points, points < newSize), axis=-1)
      )
 
      cX, cY = -startPt
      transM = np.matmul(
        np.array([[1, 0, cX],[0, 1, cY], [0, 0, 1]]),
        np.vstack([transM, [0,0,1]])
      ).astype(np.float32)[:2]
      
      res.append((points, newSize, transM, tuple(visibleInd)))
      if N <= len(res): break
      continue
    return res
  
  def apply(img, srcPoints, pointsMin, fallback=False, priorities=None, EMap=None):
    imgHW = img.shape[:2]
    newSize = imgHW[::-1]
    points = np.array(srcPoints, np.float32)
    transM = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    validPointsMask = np.array([_isVisible(pt, imgHW[::-1]) for pt in srcPoints])
    
    if not fallback:
      N = 1 if priorities is None else maxCandidates
      if not(EMap is None):
        affineTrans = EMTransformations(img, srcPoints, validPointsMask, EMap, N=N)
      else:
        affineTrans = affineTransformations(
          img, srcPoints, 
          pointsMin=pointsMin if priorities is None else -1,
          validPointsMask=validPointsMask,
          N=N,
          includeZeroPoints=not(priorities is None)
        )
      if affineTrans:
        if not(priorities is None):
          minV = min(affineTrans, key=lambda x: priorities[x[-1]])
          affineTrans = [minV]
        points, newSize, transM, _ = affineTrans[0]
    ##
    if not(resize is None):
      scales = np.divide(resize, newSize)
      points = np.array([np.array(x) * scales for x in points])
      transM = np.matmul(
        np.array([[scales[0], 0, 0],[0, scales[1], 0], [0, 0, 1]]),
        np.vstack([transM, [0,0,1]])
      ).astype(np.float32)[:2]
      newSize = np.array(resize)

    img = cv2.warpAffine(
      img, transM[:2], tuple(newSize.astype(np.uint16)),
      flags=cv2.INTER_NEAREST, borderMode=borderMode
    )
    
    if 0 < KPMaskRate:
      img, points = applyKPMasks(img, points)
       
    blur = int(motionBlurStrength())
    if 0 < blur:
      img = apply_motion_blur(img, blur, motionBlurAngle())
   
    if not (brightnessFactor is None):
      img = brightness_augment(img, factor=brightnessFactor())
     
    img = applyNoise(img)
    img, points = applyFlip(img, points)

    points[~validPointsMask] = -img.shape[0]
    if showAugmented:
      cv2.imshow('augmented', img)
    return img, points
  return apply
#########################################
DEFAULT_AUGMENTATIONS = {
  'paddingScale': (0.75, 8.0),
  'rotateAngle': (-130, +130),
  'brightnessFactor': (0.5, 1.5),
  'noiseRate': 0.03,
  'aspectRatioRange': (0.25, 4),
  
  'paddingClipping': (-0.1, 1.1),
  'noiseMode': 'RGB',
  'multiplicativeNoise': 0.1,
  'showAugmented': False,
  'minSize': 64,
  'minDistance': 0.02,
  'maxFails': 100,
  'maxCandidates': 5,
  'borderMode': cv2.BORDER_REPLICATE,
  'flipX': 0.5,
  'flipY': 0.5,
  'motionBlurStrength': (0, 10),
  'motionBlurAngle': (-180, 180),
  'KPMaskRate': 0.05,
  'KPMaskSizeRange': (3, 20),
}