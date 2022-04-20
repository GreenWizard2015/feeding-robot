import os
import json
import random
import cv2
import hashlib
import glob
import Utils
from scripts.common import PROJECT_FOLDER, loadDataset

def extractFrames(videoSrc, frames):
  fullPath = next(
    (x for x in glob.iglob(os.path.join(PROJECT_FOLDER, '**', videoSrc), recursive=True)),
    None
  )
  if fullPath is None: return

  frame = 0
  cam = cv2.VideoCapture(fullPath)
  while True:
    ret, frameImg = cam.read()
    if not ret: break
    
    if frame in frames:
      destImg = frames[frame]
      os.makedirs(os.path.dirname(destImg), exist_ok=True)
      cv2.imwrite(destImg, frameImg)
      print('Extracted %s:%s' % (videoSrc, frame))

    frame += 1
    continue
  return

class CDatasetSamples:
  ANCHORS = Utils.ANCHORS
  
  def __init__(self, dataset=None, folder='cache', samples=None):
    self._samples = []
    
    if samples is None:
      samples = Utils.flattenDataset(loadDataset() if dataset is None else dataset)
      
    missing = {}
    for name, frame, data in samples:
      frameImg = os.path.abspath(os.path.join(
        folder,
        '%s.bmp' % (hashlib.sha256((name + str(frame)).encode('utf8')).hexdigest(), )
      ))
      if not os.path.exists(frameImg):
        if not(name in missing):
          missing[name] = {}
        missing[name][int(frame)] = frameImg
        
      for points in self._readPoints(data):
        self._samples.append({
          'file': frameImg,
          'points': points,
          'raw': (name, frame, data)
        })
      continue
    
    for name, frames in missing.items():
      extractFrames(name, frames)
    
    self.caching(enabled=len(self) < 10)
    return
  
  def caching(self, enabled):
    self._memCache = {} if enabled else None
    return
  
  def _readSingleSamplePoints(self, data):
    res = []
    hasAnyUncertained = any([('confidence' in x) for x in data.items()])
    defaultConfidence = 0.0 if hasAnyUncertained else 1.0
    for anchor in self.ANCHORS:
      pt = (-1, -1, defaultConfidence)
      if anchor in data:
        x = data[anchor]
        pt = (x['x'], x['y'], x.get('confidence', defaultConfidence))
      res.append(pt)
      continue
    return res
  
  def _readPoints(self, data):
    data = [data] if isinstance(data, dict) else data
    return [self._readSingleSamplePoints(x) for x in data]
  
  def sample(self):
    return self.decode(random.choice(self._samples))
  
  def decode(self, data):
    F = data['file']
    if self._memCache is None:
      srcImage = cv2.imread(F)
    else:
      srcImage = self._memCache[F] if F in self._memCache else cv2.imread(F)
      self._memCache[F] = srcImage
      
    points = [tuple(x[:2]) for x in data['points']]
    pointsConfidence = [x[2] for x in data['points']]
    return srcImage, points, pointsConfidence
    
  @property
  def samples(self):
    return self._samples
  
  def __len__(self):
    return len(self._samples)