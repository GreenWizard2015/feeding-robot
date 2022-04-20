import numpy as np
import random
from collections import defaultdict
import json
import cv2
import copy
import os
import sys
import matplotlib.pyplot as plt

ANCHORS = ['UR', 'UY', 'UW', 'DB', 'DG', 'DW', 'B1', 'B2']

FULLSIZE_CROPS = [
  # full size
  (0, 0, 1, 1),
  # mirrored/flipped
  (0, 1, 1, 0),
  (1, 0, 0, 1),
  (1, 1, 0, 0),
]

TEST_CROPS = FULLSIZE_CROPS + [
  # center crop
  (0.3, 0.3, 0.7, 0.7),
  # quarters
  (0.0, 0.0, 0.5, 0.5),
  (0.0, 0.5, 0.5, 1.0),
  (0.5, 0.0, 1.0, 0.5),
  (0.5, 0.5, 1.0, 1.0),
  # halved
  (0.0, 0.0, 0.5, 1.0),
  (0.0, 0.0, 1.0, 0.5),
  (0.0, 0.5, 1.0, 1.0),
  (0.5, 0.0, 1.0, 1.0),
]

def resetSeed(seed_value=42):
  os.environ['PYTHONHASHSEED'] = str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  import tensorflow as tf
  tf.compat.v1.set_random_seed(seed_value)
  tf.random.set_seed(seed_value)
  return

def setupEnv(seed=42):
  sys.path.append(os.path.dirname(os.path.dirname(__file__))) # fix resolving in colab and eclipse ide
  
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
  import tensorflow as tf
  tf.get_logger().setLevel('WARNING')

  if 'COLAB_GPU' not in os.environ: # local GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
      gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2 * 1024)]
    )
  
  if not (seed is None):
    resetSeed(seed)
  return

def fixedDatasetFrom(dataset, preprocess, crops, gaussian):
  from CFixedDataLoader import CFixedDataLoader
  return CFixedDataLoader(dataset, preprocess, crops, gaussian)

def takeSample(gen, batch=0, sampleID=0):
  (X, conf), (Y, masks) = gen[batch]
  res = ((X[sampleID], conf[sampleID]), (Y[sampleID], masks[sampleID]))
  del X # just to be sure
  del masks # just to be sure
  return res

def flattenDataset(dataset):
  samples = []
  for v, d in dataset.items():
    for f, data in d.items():
      data = data if isinstance(data, list) else [data]
      for x in data:
        samples.append((v, f, x))
  return samples

def samples2dataset(samples):
  res = {}
  for v, f, data in samples:
    if not(v in res):
      res[v] = {}
      
    if f in res[v]:
      old = res[v][f]
      data = old + [data] if isinstance(old, list) else [old, data]

    res[v][f] = data
    continue
  return res

def confidence(sample):
  hasAnyUncertained = any([('confidence' in x) for x in sample.values()])
  if not hasAnyUncertained: return 1.0
  
  confidence = np.zeros(8, np.float)
  for i, x in enumerate(sample.values()):
    confidence[i] = x['confidence']
  return np.mean(confidence)

def sortedSamples(samples):
  return list(sorted(samples, key=lambda x: confidence(x[-1]), reverse=True))

def dropPoints(samples, threshold):
  res = []
  for sample in samples:
    A, B, points = sample
    points = {
      name: data for name, data in points.items()
      if threshold[ANCHORS.index(name)] < data.get('confidence', 1)
    }
    if points:
      res.append((A, B, points))
    continue
  return res

def normalizeConfidence(samples):
  maxS = np.zeros((8,), np.float32)
  for _, _, points in samples:
    for name, data in points.items():
      ind = ANCHORS.index(name)
      maxS[ind] = max((maxS[ind], data.get('confidence', 1)))
      continue
    continue
  
  res = []
  for A, B, points in samples:
    pts = {
      name: {
        'x': data['x'],
        'y': data['y'],
        'confidence': (data.get('confidence', 1) / maxS[ANCHORS.index(name)]) ** 2
      }
      for name, data in points.items()
    }
    res.append((A, B, pts))
    continue
  return res

def samplesBy(dataset, by='random'):
  samples = flattenDataset(dataset)
  
  if 'random' == by:
    random.shuffle(samples)
  if 'confidence' == by:
    samples = sortedSamples(samples)
  return samples

def datasetLen(dataset):
  return len(flattenDataset(dataset))

def limitDataset(dataset, N, by='random'):
  samples = samplesBy(dataset, by)
  return samples2dataset(samples[:N])

def splitDatasetAt(dataset, position, by='random'):
  samples = samplesBy(dataset, by)  
  assert 1 < len(samples), 'Dataset must contain more than one sample.'
   
  left = samples[:position]
  right = samples[position:]
  assert (0 < len(left)) and (0 < len(right))
  return samples2dataset(left), samples2dataset(right)

def splitDataset(dataset, fraction, by='random'):
  samples = samplesBy(dataset, by)
  leftFraction = int(fraction * len(samples))
  if len(samples) - leftFraction < 1:
    leftFraction = len(samples) - 1
  
  return splitDatasetAt(dataset, position=leftFraction, by=by)

def readVideo(filename):
  cam = cv2.VideoCapture(filename)
  try:
    while True:
      ret, img = cam.read()
      if not ret: break
      yield img
  finally:
    cam.release()
  return

def readAllFrames():
  from scripts.common import PROJECT_FOLDER, SOURCE_FILES
  # loop through ALL video files
  for videoFile in SOURCE_FILES:
    videoSrc = os.path.basename(videoFile)
    def frames():
      for frameID, img in enumerate(readVideo(videoFile)):
        yield(videoSrc, frameID, img)
      return
    
    yield(videoSrc, frames)
  return
  
def extractSamples(detect, SKIP_AFTER_DETECTION=5, minFrame=0):
  res = []
  for videoSrc, frames in readAllFrames():
    print('Processing "%s"' % (videoSrc))
    
    lastD = -SKIP_AFTER_DETECTION
    for _, frameID, img in frames():
      if frameID < minFrame: continue
      if (frameID - lastD) < SKIP_AFTER_DETECTION: continue
      
      # apply detector to frame and save predictions
      points = detect(img)
      points = {k: v for k, v in points.items() if not(v is None)}
      if points:
        res.append((videoSrc, str(frameID), points))
        lastD = frameID
      continue
    continue
  
  return res

def datasetInfo(dataset):
  samples = flattenDataset(dataset)
  print('Total samples: %d' % (len(samples), ))
  bySource = defaultdict(int)
  for video, _, _ in samples:
    bySource[video] += 1
  
  for video, N in bySource.items():
    print('%s: %d' % (video, N))
  return


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save(filename, data):
  if filename.lower().endswith('.json'):
    with open(filename, 'w') as f:
      json.dump(data, f, indent=2, cls=NumpyEncoder)
    return
  raise Exception('Unknown format. (%s)' % filename)

def saveMetrics(metrics, filepath, startEpoch=0):
  collectedData = defaultdict(dict)
  for dataName, values in metrics.items():
    name = dataName.replace('val_', '')
    metricKind = 'test' if dataName.startswith('val_') else 'train'
    collectedData[name][metricKind] = list(values)
    
  for name, data in collectedData.items():
    plt.clf()
    fig = plt.figure()
    axe = fig.subplots(ncols=1, nrows=1)
    for nm, values in data.items():
      axe.plot(values[startEpoch:], label=nm)
      
    axe.title.set_text(name)
    axe.set_ylabel(name)
    axe.set_xlabel('epoch')
    axe.legend(loc='upper left')
    fig.savefig(filepath('%s.png' % name))
    plt.close(fig)
    
  # dump metrics
  save(filepath('history.json'), metrics)
  return

def gaussian(size, fwhm):
  paddedSz = int(size + fwhm)
  x = np.arange(0, paddedSz * 2, 1, np.float32)
  y = x[:,np.newaxis]
  centralGaussian = np.exp(-((x-paddedSz)**2 + (y-paddedSz)**2) / fwhm**2)

  def f(center):
    dx, dy = np.subtract([paddedSz, paddedSz], center)
    if (0 <= dx <= paddedSz) and (0 <= dy <= paddedSz):
      return centralGaussian[dx:dx+size, dy:dy+size]
    return np.zeros((size, size), np.float32)
  return f
