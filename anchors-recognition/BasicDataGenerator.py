import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
import json
import random
import joblib

def gaussian(size, fwhm):
  x = np.arange(0, size * 2, 1, float)
  y = x[:,np.newaxis]
  centralGaussian = np.exp(-4*np.log(2) * ((x-size)**2 + (y-size)**2) / fwhm**2)

  def f(center):
    dx, dy = np.subtract([size, size], center)
    return centralGaussian[dy:dy+size, dx:dx+size]
  return f

class CBasicDataGenerator(Sequence):
  ANCHORS = ['UR', 'UY', 'UW', 'DB', 'DG', 'DW', 'B1', 'B2']
  
  def __init__(self, params):
    self._batchSize = params['batch size']
    self._batchesPerEpoch = params['batches per epoch']
    self._dims = np.array(params['image size'])
    self._transformer = params['transformer']
    self._minAnchors = params['min anchors']
    self._preprocess = params.get('preprocess', lambda x: x)
    self._addFakeOutput = params.get('fake output', False)
    self._gaussian = gaussian(
      size=self._dims[0],
      fwhm=params['target radius']
    )
    
    self._executor = lambda x: list(x)
    self._createSample = self._createSampleReal
    if 1 < joblib.cpu_count():
      self._executor = joblib.Parallel(n_jobs=-1, require='sharedmem')
      self._createSample = joblib.delayed(self._createSampleReal)
    
    FOLDER = os.path.abspath(params['folder'])
    _file = lambda name: os.path.join(FOLDER, name)
    
    with open(_file('samples.json')) as f:
      self._samples = [
        { 'file': _file('%s.bmp' % name), 'points': self._readPoints(data) }
        for name, data in json.load(f).items()
      ]

    self._epochBatches = None
    self.on_epoch_end()
    return 

  def __len__(self):
    return self._batchesPerEpoch

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._epochBatches = random.choices(self._samples, k=self._batchesPerEpoch)
    return
  
  def _visiblePoint(self, pt):
    x, y = pt
    return not( (x < 0) or (self._dims[0] <= x) or (y < 0) or (self._dims[1] <= y) )

  def _countVisible(self, points):
    return sum([1 for pt in points if self._visiblePoint(pt)])
  
  def _createSampleReal(self, srcImage, points, X, Y, ind):
    pointsMin = min((
      self._minAnchors,
      sum(1 for x, y in points if (0 <= x) and (0 <= y))
    ))
    while True:
      transformed = self._transformer(srcImage, points)
      if not transformed: continue
      
      img, pts = transformed
      scales = (self._dims / np.array(img.shape[:2], np.float32))[::-1]
      pts = [np.array(x) * scales for x in pts]
      
      if self._countVisible(pts) < pointsMin: continue
      
      noneClassMasks = np.ones((*self._dims, ), np.float32)
      for i, pt in enumerate(pts):
        mask = None
        if self._visiblePoint(pt):
          x, y = pt
          mask = self._gaussian(center=(int(x), int(y)))
          noneClassMasks -= mask
        else:
          mask = np.zeros_like(noneClassMasks)
        
        Y[ind, :, :, i] = mask
      #######
      Y[ind, :, :, -1] = noneClassMasks
      X[ind] = self._preprocess(img)
      break
    return
  
  def __getitem__(self, index):
    data = self._epochBatches[index]
    srcImage = cv2.imread(data['file'])
    points = data['points']
    
    X = np.empty((self._batchSize, *self._dims, 3), np.float32)
    Y = np.empty((self._batchSize, *self._dims, len(self.ANCHORS) + 1), np.float32)
    
    self._executor(
      self._createSample(srcImage, points, X, Y, ind) for ind in range(self._batchSize)
    )

    Y[:, :, :, -1] = np.clip(Y[:, :, :, -1], a_min=0., a_max=1.)

    if self._addFakeOutput:
      Y = [Y, np.zeros((Y.shape[0] * 9, 1))]
    return (X, Y)
  ###########################
  def _readPoints(self, data):
    res = []
    for anchor in self.ANCHORS:
      pt = (-1, -1)
      if anchor in data:
        pt = (data[anchor]['x'], data[anchor]['y'])
      res.append(pt)
    return res