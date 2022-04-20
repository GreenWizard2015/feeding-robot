import numpy as np
from tensorflow.keras.utils import Sequence
import random
import math
import Utils
from collections import defaultdict

def emptyTransformer(srcImage, points, pointsMin, fallback):
  return srcImage, points

class CDataLoader(Sequence):
  def __init__(self, samples, params):
    self._samples = samples
    self._batchSize = params['batch size']
    self._preprocess = params['preprocess']
    self._maxFails = params.get('max fails', 20)
    self._transformer = params.get('transformer', emptyTransformer)
    self._gaussian = params['gaussian']
    
    self._priority = defaultdict(int) if params.get('prioritized sampling', False) else None
    minAnchors = params.get('min anchors', (0, 8))
    if not isinstance(minAnchors, (tuple, list)):
      minAnchors = (minAnchors, minAnchors)
    self._minAnchors = minAnchors

    N = len(samples.samples)
    samplesPerEpoch = params.get('samples per epoch', -1)
    if samplesPerEpoch <= 0:
      augmentationsPerSample = params.get('augmentations per sample', 1)
      samplesPerEpoch = math.ceil((N * augmentationsPerSample) / self._batchSize) * self._batchSize
    
    assert (samplesPerEpoch % self._batchSize) == 0
    assert N <= samplesPerEpoch
    self._samplesIndices = np.arange(samplesPerEpoch) % N
    self._EMap = {}
    
    self.stats = {}
    self.on_epoch_end()
    return

  def __len__(self):
    return len(self._samplesIndices) // self._batchSize

  def on_epoch_end(self):
    np.random.shuffle(self._samplesIndices)
    return
  
  def _visiblePoint(self, pt):
    x, y = pt
    return (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)

  def _countVisible(self, points):
    return sum([1 for pt in points if self._visiblePoint(pt)])
  
  def _pointsMinSelector(self, pointsN):
    def f():
      res = random.randint(*self._minAnchors)
      return min((res, pointsN))
    return f
  
  def _createSample(self, index):
    srcImage, points, confidence = self._samples.decode(self._samples.samples[index])
    
    validPoints = np.array([(0 <= x) and (0 <= y) for x, y in points])
    pointsMinSelector = self._pointsMinSelector(validPoints.sum())
    
    tries = self._maxFails
    while True:
      tries -= 1
      fallback = tries < 1
      
      pointsMin = 0 if fallback else pointsMinSelector()
      sample = self._transformer(
        srcImage, points, pointsMin, fallback=fallback,
        priorities=self._priority, EMap=self._EMap.get(index, None)
      )
      if sample is None: continue
      
      img, pts = sample
      YCoords = pts[:, ::-1] / img.shape[:2] # px to ratio
      YCoords[~validPoints] = -1
      
      if self._countVisible(YCoords) < pointsMin: continue

      self._stats(index, YCoords, fallback=fallback)
      return(self._preprocess(img), confidence, YCoords)
    raise Exception('Oops.....')
  
  def _stats(self, index, points, fallback=False):
    index = int(index)
    stats = self.stats.get(index, False)
    if not stats:
      self.stats[index] = stats = {
        'fallback': 0,
        'points': [0] * len(points),
        'points in sample': [0] * (1 + len(points)),
      }
    
    if fallback:
      stats['fallback'] += 1
    
    visibleN = 0
    pattern = []
    for i, pt in enumerate(points):
      visible = self._visiblePoint(pt)
      if visible:
        visibleN += 1
        stats['points'][i] += 1
        pattern.append(i)
      continue
    
    if not(self._priority is None):
      self._priority[tuple(pattern)] += 1
    stats['points in sample'][visibleN] += 1
    return
  
  def __getitem__(self, index):
    X = []
    confidence = []
    YCoords = []
    YMasks = []
    
    indices = self._samplesIndices[index*self._batchSize:(index+1)*self._batchSize]
    for sampleInd in indices:
      img, conf, coords = self._createSample(sampleInd)
      X.append(img)
      confidence.append(conf)
      YCoords.append(coords)
      
      HW = img.shape[0]
      masks = [self._gaussian((int(x * HW), int(y * HW))) for x, y in coords]
      YMasks.append(np.transpose(masks, (1, 2, 0)))
      continue

    return (
      [np.array(X, np.float32), np.array(confidence, np.float32)],
      [np.array(YCoords, np.float32), np.array(YMasks, np.float32)]
    )

  def updateEMap(self, provider, batchSize=16):
    self._EMap.clear()
    samples = self._samples.samples
    for BInd in range(0, len(samples), batchSize):
      batchSamples = samples[BInd:BInd+batchSize]
      X = []
      Y = []
      for sample in batchSamples:
        srcImage, points, _ = self._samples.decode(sample)
        points = np.array(points, np.float32)
        validPoints = np.array([(0 <= x) and (0 <= y) for x, y in points])
        coords = np.divide(points[:, ::-1], srcImage.shape[:2])
        coords[~validPoints] = -1
      
        img = self._preprocess(srcImage)
        X.append(img)
        HW = img.shape[0]
        Y.append(np.transpose(
          [self._gaussian((int(x * HW), int(y * HW))) for x, y in coords],
          (1, 2, 0)
        ))
        continue
      
      EMaps = provider(np.array(X, np.float32), np.array(Y, np.float32))
      for ind, emap in zip(np.arange(BInd, BInd + batchSize, 1), EMaps):
        self._EMap[ind] = emap
      continue
    return