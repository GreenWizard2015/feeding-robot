import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
import json
import random

def makeGaussian(size, fwhm, center):
  """ Make a square gaussian kernel.
  size is the length of a side of the square
  fwhm is full-width-half-maximum, which
  can be thought of as an effective radius.
  """
  x = np.arange(0, size, 1, float)
  y = x[:,np.newaxis]
  
  if center is None:
    x0 = y0 = size // 2
  else:
    x0 = center[0]
    y0 = center[1]
  
  return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class CBasicDataGenerator(Sequence):
  ANCHORS = ['UR', 'UY', 'UW', 'DB', 'DG', 'DW', 'B1', 'B2']
  
  def __init__(self, params):
    self._batchSize = params['batch size']
    self._batchesPerEpoch = params['batches per epoch']
    self._dims = np.array(params['image size'])
    self._transformer = params['transformer']
    self._minAnchors = params['min anchors']
    self._targetRadius = params['target radius']
    self._preprocess = params.get('preprocess', lambda x: x)
    
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
  
  def __getitem__(self, index):
    data = self._epochBatches[index]
    srcImage = cv2.imread(data['file'])
    points = data['points']
    initPointsN = self._countVisible(points)
    
    X = np.zeros((self._batchSize, *self._dims, 3), np.float32)
    Y = np.zeros((self._batchSize, *self._dims, len(self.ANCHORS) + 1), np.float32)
    resIndex = 0
    while resIndex < self._batchSize:
      transformed = self._transformer(srcImage.copy(), points)
      if transformed:
        img, pts = transformed
        scales = (self._dims / np.array(img.shape[:2], np.float32))[::-1]
        pts = [np.array(x) * scales for x in pts]
        pts = [(pt if self._visiblePoint(pt) else (-1, -1)) for pt in pts]
        
        if self._countVisible(pts) < min((self._minAnchors, initPointsN)):
          #print(data)
          continue
        
        noneClassMasks = np.ones((*self._dims, ), np.float32)
        for i, pt in enumerate(pts):
          mask = None
          if self._visiblePoint(pt):
            x, y = pt
            mask = makeGaussian(self._dims[0], fwhm=self._targetRadius, center=(int(x), int(y)))
            noneClassMasks -= mask
          else:
            mask = np.zeros_like(noneClassMasks)
          
          Y[resIndex, :, :, i] = mask
          #####
        Y[resIndex, :, :, -1] = noneClassMasks
        Y[resIndex] = np.clip(Y[resIndex],  a_min=0., a_max=1.)
        
        X[resIndex] = self._preprocess( cv2.resize(img, tuple(int(x) for x in self._dims) ) )
        resIndex += 1
      ###

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