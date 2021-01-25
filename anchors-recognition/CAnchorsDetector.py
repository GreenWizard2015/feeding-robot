import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import numpy as np
import cv2
from scipy.ndimage.measurements import center_of_mass

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu")(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def downsamplingBlockWithLink(prev, sz, filters):
  link = convBlock(prev, sz, filters)
  
  res = link
  for _ in range(3):
    res = convBlock(res, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same")(res)
  return link, res
  
def upsamplingBlock(prev, shortcut, sz, filters):
  prev = layers.Convolution2DTranspose(filters, (2, 2), strides=2)(prev)
  concatenated = layers.Concatenate()([prev, shortcut])
   
  return convBlock(concatenated, sz, filters)

def anchorsDetectorModel(size, anchors):
  res = inputs = layers.Input(shape=(*size, 3))

  convA, res = downsamplingBlockWithLink(res, 3, 64)
  convB, res = downsamplingBlockWithLink(res, 5, 64)
  convC, res = downsamplingBlockWithLink(res, 5, 64)
  convD, res = downsamplingBlockWithLink(res, 7, 64)

  res = convBlock(res, 7, 128)

  res = upsamplingBlock(res, convD, 7, 64)
  res = upsamplingBlock(res, convC, 5, 64)
  res = upsamplingBlock(res, convB, 5, 64)
  res = upsamplingBlock(res, convA, 3, 64)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Conv2D(1 + anchors, 1, activation='softmax', padding='same')(res)
  )
  
class CAnchorsDetector:
  def __init__(self, size=(224, 224), anchors=8, minProbability=.1):
    self._size = np.array(size)
    self._minProbability = minProbability
    self._anchors = anchors
    self._model = anchorsDetectorModel(size, anchors)
    self._pointRadius = 10
    return
  
  def preprocess(self, img):
    if not np.array_equal(np.array(img.shape[:2]), self._size):
      img = cv2.resize(img, tuple(self._size))
    
    return img / 255.
  
  @property
  def network(self):
    return self._model
  
  def weights_file(self, kind):
    return os.path.join(os.path.dirname(__file__), 'anchors-detector-%s.h5' % kind)
  
  def load(self, kind=''):
    return self._model.load_weights(self.weights_file(kind)) 

  def _decodePrediction(self, prediction, image, returnProbabilities):
    res = [None for _ in range(self._anchors)]
    invScale = self._size / image.shape[:2]
    
    X = np.arange(self._size[0])[None, :] if returnProbabilities else None
    Y = np.arange(self._size[0])[:, None] if returnProbabilities else None
    
    for ind in range(self._anchors):
      pred = prediction[:, :, ind]
      maxProb = pred.max()
      if self._minProbability < maxProb:
        centerPt = center_of_mass(pred)
        if returnProbabilities:
          cy, cx = centerPt
          mask = (X - cx) ** 2 + (Y - cy) ** 2 < self._pointRadius ** 2
          predArea = pred[mask].reshape(-1)
          res[ind] = (centerPt / invScale, maxProb, predArea)
        else:
          res[ind] = centerPt / invScale
      #
    return res
  
  def _predict(self, images, returnProbabilities):
    predicted = self._model.predict(np.array([
      self.preprocess(img) for img in images
    ]))
    
    return [self._decodePrediction(predicted[i], img, returnProbabilities) for i, img in enumerate(images)]
  
  def detect(self, img, returnProbabilities=False):
    return self._predict([img], returnProbabilities)[0]
  
  def combinedDetections(self, image):
    images = [image]
    
    iw, ih = image.shape[:2]
    w, h = (np.array(image.shape[:2]) / 1.5).astype(np.int)
    images.append(image[:w, :h])
    images.append(image[-w:, :h])
    images.append(image[-w:, -h:])
    images.append(image[:w, -h:])
    images.append(image[w//2:w+w//2, h//2:h+h//2])
    shifts = np.array( [(0, 0), (0, 0), (iw - w, 0), (iw - w, ih - h), (0, ih - h), (w//2, h//2)] )
    
    predictions = self._predict(images, returnProbabilities=True)
    
    predictedAnchors = [[] for _ in range(self._anchors)]
    for anchors, shift in zip(predictions, shifts):
      for i, x in enumerate(anchors):
        if not (x is None):
          pt, _, areaProb = x
          prob = areaProb.max() if 0 < areaProb.size else 0
          predictedAnchors[i].append(( np.array(pt) + shift, prob ))
    ###########
    # combine anchors
    res = [None for _ in range(self._anchors)]
    for i, predAnchor in enumerate(predictedAnchors):
      if len(predAnchor) <= 0: continue

      # TODO: suppress outliers
      pts = np.array([x[0] for x in predAnchor])
      probs = np.array([x[1] for x in predAnchor]) + .0001
      meanPt = np.average(pts, weights=probs, axis=0)
      res[i] = (meanPt, probs.mean())
    return res