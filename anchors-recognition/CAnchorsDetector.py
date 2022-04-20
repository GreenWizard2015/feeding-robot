import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import numpy as np
import cv2
from HeatmapDecodingLayer import HeatmapDecodingLayer
import gc
import glob
from PatchCombiners import PatchCombinersMapping
from CropsGenerators import CropsGeneratorsMapping
import hashlib
import json
from Utils import ANCHORS

def _featuresProvider(shape):
  base_model = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False)

  # Use the activations of these layers
  layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
  ]
  layers = [base_model.get_layer(name).output for name in layer_names]
  
  # Create the feature extraction model
  model = tf.keras.Model(inputs=base_model.input, outputs=layers)
  model.trainable = False
  return model

# from tensorflow_examples.models.pix2pix import pix2pix
def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())
  return result

def anchorsDetectorModel(size, anchors, asSegmentation=False, segmap=False):
  Resizing = layers.experimental.preprocessing.Resizing
  inputs = layers.Input(shape=(*size, 3))
  sharpness = tf.Variable(1.0)
  inputImg = inputs

  featuresNet = _featuresProvider(shape=(*size, 3))
  features = featuresNet((2.0 * inputImg) - 1.0)[::-1]

  res = None
  lastBlock = [False for _ in features[:-1]] + [True]
  for FV, isLast in zip(features, lastBlock):
    H, W, _ = FV.shape[1:]
    resized = Resizing(H, W)(inputImg)
    
    parts = [resized, FV]
    if not(res is None):
      parts.append(res)
    
    concatenated = res = layers.Concatenate(axis=-1)(parts)
    if not isLast:
      res = upsample(32, 3, apply_dropout=True)(concatenated)
    continue

  # DON'T use BatchNorm/Conv2DTranspose due to chessboard pattern
  res = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(res)
  res = layers.UpSampling2D()(res)
  res = layers.Dropout(0.25)(res)
  
  res = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(res)
  if asSegmentation:
    res = segL = layers.Lambda(lambda x: x[..., :anchors])(tf.nn.softmax(res))
  
  heatmaps = layers.Conv2D(
    anchors, 1, activation='relu', padding='same',
    kernel_initializer=tf.initializers.random_normal(0.0, 0.01)
  )(res)
  
  decodedPos = HeatmapDecodingLayer()(heatmaps)

  res = [decodedPos, heatmaps]
  if asSegmentation and segmap:
    res.append(
      layers.Lambda(lambda x: tf.argmax(x, axis=-1))(segL)
    )
  return keras.Model(inputs=inputs, outputs=res, name='ADet')

def combineHeatmaps(heatmaps, returnHeatmaps=False, raw=False):
  res = [] if raw else {name: None for name in ANCHORS}
  for heatmap, name in zip(heatmaps, ANCHORS):
    x, y = np.unravel_index(heatmap.argmax(), heatmap.shape)
    prob = float(heatmap[x, y])
    if raw:
      res.append(((x, y), prob))
      continue
    
    res[name] = {
      'x': int(y), 'y': int(x),
      'confidence': prob
    }
    continue
  
  if returnHeatmaps:
    return res, heatmaps
  return res

class CAnchorsDetector:
  def __init__(self, size=(224, 224), anchors=8, networkOpts={}):
    self._size = np.array(size)
    self._anchors = anchors
    self._model = anchorsDetectorModel(size, anchors, **networkOpts)
    return

  @staticmethod
  def preprocessImage(img, size):
    if not np.array_equal(np.array(img.shape[:2]), size):
      img = cv2.resize(img, tuple(size))
    
    return img / 255.
  
  def preprocess(self, img):
    return self.preprocessImage(img, self._size)
  
  @property
  def network(self):
    return self._model
  
  def weights_file(self, kind, folder=None):
    return os.path.join(
      os.path.dirname(__file__) if folder is None else folder,
      'anchors-detector-%s.h5' % kind
    )
  
  @staticmethod
  def models(folder=None):
    folder = os.path.dirname(__file__) if folder is None else folder
    folder = os.path.abspath(folder)
    return glob.glob(os.path.join(folder, 'anchors-detector-*.h5'))
  
  def load(self, kind='', folder=None, file=None):
    if not file:
      file = self.weights_file(kind, folder)
    self._model.load_weights(file)
    return self
  
  def save(self, kind='', folder=None):
    weights_file = self.weights_file(kind, folder)
    os.makedirs(os.path.dirname(weights_file), exist_ok=True)
    return self._model.save_weights(weights_file) 

  def _decodePrediction(self, coords, heatmaps, image):
    res = []
    for pt in coords:
      prob = pt[-1]
      pt = np.multiply(pt[:2], image.shape[:2])
      res.append((pt, prob))
    return res

  @tf.function
  def _penalize(self, x, penalize):
    if not penalize: return x
    
    meanX = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    return x - meanX
    
  @tf.function
  def _TTAFlip(self, images, penalize):
    x, res = self._model(images)
    res = 1.0 + self._penalize(res, penalize)
    for dx, dy in [(1, -1), (-1, 1), (-1, -1)]:
      _, M = self._model(images[:, ::dx, ::dy, :])
      res = res * (1.0 + self._penalize(M[:, ::dx, ::dy, :], penalize))
      continue
    return x, tf.experimental.numpy.log2(res) / 4.0

  @tf.function
  def _TTANoise(self, images, N, stddev, useFlip, penalize):
    x, res = self._TTAFlip(images, penalize) if useFlip else self._model(images)
    res = 1.0 + self._penalize(res, penalize)
    for _ in tf.range(1, N):
      I = images * tf.random.normal(tf.shape(images), mean=1.0, stddev=stddev, dtype=images.dtype)
      I = tf.clip_by_value(I, 0.0, 1.0)
      
      _, B = self._TTAFlip(I, penalize) if useFlip else self._model(I)
      res = res * (1.0 + self._penalize(B, penalize))
      continue
    return x, tf.experimental.numpy.log2(res) / float(N)
  
  def _TTA(self, images, TTA):
    mode = TTA.get('mode', 'flip').lower() if isinstance(TTA, dict) else 'flip'
    penalize = isinstance(TTA, dict) and TTA.get('penalize', False)
    if 'noise' == mode:
      return self._TTANoise(
        images,
        N=TTA.get('N', 8), stddev=TTA.get('std', 0.1),
        useFlip=TTA.get('use flip', False),
        penalize=penalize
      )
    if 'flip' == mode:
      return self._TTAFlip(images, penalize=penalize)
    return None
  
  def _infer(self, images, TTA=True):
    inputs = np.array([ self.preprocess(img) for img in images ])
    if TTA:
      coords, heatmaps = self._TTA(inputs, TTA)
    else:
      coords, heatmaps = self._model.predict(inputs)
    del inputs
    return coords.numpy(), heatmaps.numpy()
  
  def _predict(self, images, TTA=True):
    coords, heatmaps = self._infer(images, TTA)
    return [
      self._decodePrediction(coords[i], heatmaps[i], img)
      for i, img in enumerate(images)
    ]
  
  def detect(self, img, raw=False, TTA=False):
    predictedAnchors = self._predict([img], TTA)[0]
    if raw: return predictedAnchors
    
    res = {name: None for name in ANCHORS}
    for i, predAnchor in enumerate(predictedAnchors):
      if predAnchor is None: continue
      (y, x), prob = predAnchor
      res[ANCHORS[i]] = {
        'x': int(x), 'y': int(y),
        'confidence': min((0.99, int(prob * 1000) / 1000))
      }
    return res 
  
  def _cropsForHeatmaps(self, image, crops):
    crops = CropsGeneratorsMapping.get(crops, crops)
    crops = crops(image.shape[:2], self._size[0])
    return list(set(crops))
  
  def heatmaps(self, image, crops=None, batchSize=8, mode='avg', TTA=True):
    crops = self._cropsForHeatmaps(image, crops)
    dims = image.shape[:2]
    channels = len(ANCHORS)
    mode = mode if isinstance(mode, list) else [mode]
    combiners = [PatchCombinersMapping[name](channels, dims) for name in mode]

    for chunkI in range(0, len(crops), batchSize):
      sCrops = crops[chunkI:chunkI+batchSize]
      images = [ image[x1:x2, y1:y2] for x1, y1, x2, y2 in sCrops ]
      _, heatmaps = self._infer(images, TTA)
      del images
      heatmaps = heatmaps.transpose((0, 3, 1, 2)) # BWHC => BCWH

      for crop, anchors in zip(sCrops, heatmaps):
        x1, y1, x2, y2 = crop
        rescaled = [
          cv2.resize(nhmap, (y2 - y1, x2 - x1), interpolation=cv2.INTER_AREA)
          for nhmap in anchors
        ]
        for c in combiners:
          c.update(crop, rescaled)
      continue
    
    res = [c.result() for c in combiners]
    gc.collect()
    return res
  
  def combinedDetections(
    self, image,
    batchSize=8, crops=None, returnHeatmaps=False, raw=False, mode='avg', TTA=True
  ):
    results = [
      combineHeatmaps(heatmaps, returnHeatmaps=returnHeatmaps, raw=raw)
      for heatmaps in self.heatmaps(image, crops, batchSize, mode, TTA)
    ]
    return results if isinstance(mode, list) else results[0]
  
  def hash(self):
    H = hashlib.sha512()
    for l in self.network.get_weights():
      H.update(l)
    return H.digest()
  
class CStackedDetector:
  def __init__(self, detectors, cache=None):
    self._detectors = detectors
    self._hashes = [hashlib.md5(x.hash()).hexdigest() for x in detectors]
    self._cache = cache
    return
  
  def _imageUid(self, image, crops, mode, TTA):
    if not self._cache: return None
    imageUID = hashlib.sha256()
    imageUID.update(image)
    params = [str(x) for x in (crops, mode, TTA, image.sum(), image.mean())]
    imageUID.update(json.dumps(params).encode())
    return imageUID.hexdigest()
  
  def _heatmapsFrom(self, detectorInd, inferHeatmaps, imageUID):
    if not self._cache: return inferHeatmaps(detectorInd)
  
    filename = os.path.join(self._cache, self._hashes[detectorInd], imageUID + '.npy')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
      heatmaps = np.load(filename)
      # unpack
      heatmaps = heatmaps.astype(np.float16) / 255.0
    else:
      heatmaps = np.array(inferHeatmaps(detectorInd)).astype(np.float16)
      # pack
      np.save(filename, (heatmaps * 255.0).astype(np.uint8))

    return heatmaps
  
  def heatmaps(self, image, crops=None, batchSize=8, mode='avg', combineMode='avg', TTA=True):
    imageUID = self._imageUid(image, crops, mode, TTA)
    inferHeatmaps = lambda DInd: self._detectors[DInd].heatmaps(image, crops, batchSize, mode, TTA)[0]

    dims = image.shape[:2]
    channels = len(ANCHORS)
    combiner = PatchCombinersMapping[combineMode](channels, dims)

    for detectorInd, _ in enumerate(self._detectors):
      heatmaps = self._heatmapsFrom(detectorInd, inferHeatmaps, imageUID)
      combiner.update((0, 0, dims[0], dims[1]), heatmaps)
      continue
    
    res = combiner.result()
    gc.collect()
    return res

  def combinedDetections(
    self, image,
    batchSize=8, crops=None, returnHeatmaps=False, raw=False, mode='avg', combineMode='prod',
    TTA=True
  ):
    modes = mode if isinstance(mode, list) else [mode]
    results = [
      combineHeatmaps(
        self.heatmaps(image, crops, batchSize, mode, combineMode, TTA),
        returnHeatmaps=returnHeatmaps, raw=raw
      ) for mode in modes
    ]
    return results if isinstance(mode, list) else results[0]