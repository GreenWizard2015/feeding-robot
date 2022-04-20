import numpy as np
import cv2

def _crop(imgWH, cropsWH, shift=(0, 0)):
  cropW, cropH = cropsWH
  W, H = imgWH
  res = []
  for x in range(shift[0], W, cropW):
    for y in range(shift[0], H, cropH):
      x2 = min((x + cropW, W))
      y2 = min((y + cropH, H))
      x1 = x2 - cropW
      y1 = y2 - cropH
      if (x1 < 0) or (y1 < 0): continue
      res.append((x1, y1, x2, y2))
  return res
######################
class CcgFull:
  def __call__(self, imgWH, baseCropSize):
    return [(0, 0, *imgWH)]
  
  def __str__(self):
    return 'full'
  
class CcgDefault:
  def __call__(self, imgWH, baseCropSize):
    iw, ih = imgWH
    crops = [(0, 0, *imgWH)]
    crops.extend(_crop(imgWH, cropsWH=(int(0.75 * iw), int(0.75 * ih))))
    crops.extend(_crop(imgWH, cropsWH=(int(0.5 * iw), int(0.5 * ih))))
    
    ratio = imgWH[0] / float(imgWH[1])
    for ratio in [ratio, 1.0 / ratio]:
      for sizeScale in [1, 3]:
        szW = int(sizeScale * baseCropSize)
        cropsWH = np.array((szW, szW * ratio)).astype(np.int)
        hw, hh = cropsWH // 2
        
        crops.extend(_crop(imgWH, cropsWH=cropsWH, shift=(0, 0)))
        crops.extend(_crop(imgWH, cropsWH=cropsWH, shift=(0, hh)))
        crops.extend(_crop(imgWH, cropsWH=cropsWH, shift=(hw, 0)))
        crops.extend(_crop(imgWH, cropsWH=cropsWH, shift=(hw, hh)))
        continue
    return crops
  
  def __str__(self):
    return 'default'
  
class CcgAssertOverlapping:
  def __init__(self, minOverlapping, generator):
    self._minOverlapping = minOverlapping
    self._generator = generator
    return
  
  def __call__(self, imgWH, baseCropSize):
    cnt = np.zeros(imgWH, np.uint16)
    res = set(self._generator(imgWH, baseCropSize))
    for x1, y1, x2, y2 in res:
      cnt[x1:x2, y1:y2] += 1
    
    minCnt = cnt.min()
    assert self._minOverlapping <= minCnt, 'Must be at lease %d overlaps.' % (self._minOverlapping, )
    return res
  
  def __str__(self):
    return 'AssertOverlapping(%d, %s)' % (self._minOverlapping, str(self._generator))
######################
class CcgCombine:
  def __init__(self, *generators):
    self._generators = generators
    return
  
  def __call__(self, imgWH, baseCropSize):
    res = []
    for g in self._generators:
      res.extend(g(imgWH, baseCropSize))
      
    return res
  
  def __str__(self):
    return 'Combine(%s)' % (', '.join([str(g) for g in self._generators]),)
######################
class CcgDense:
  def __init__(self, step):
    self._step = step
    self._default = CcgDefault()
    return
  
  def __call__(self, imgWH, baseCropSize):
    crops = self._default(imgWH, baseCropSize)
    
    ratio = imgWH[0] / float(imgWH[1])
    for ratio in [ratio, 1.0 / ratio]:
      for sizeScale in [1, 3]:
        szW = int(sizeScale * baseCropSize)
        cropsWH = np.array((szW, szW * ratio)).astype(np.int)
        
        for sx in range(0, cropsWH[0], self._step):
          for sy in range(0, cropsWH[1], self._step):
            crops.extend(_crop(imgWH, cropsWH=cropsWH, shift=(sx, sy)))
        continue
    return crops
  
  def __str__(self):
    return 'dense(%d)' % (self._step)
  
######################
CropsGeneratorsMapping = {
  'default': lambda *a: CcgDefault()(*a),
  'all': lambda *a: CcgDefault()(*a),
  None: lambda *a: CcgDefault()(*a),
  
  'full': lambda *a: CcgFull()(*a),
  
  'dense': lambda *a: CcgDense(step=12)(*a),
}