from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import Utils

def _cropOf(image, points, crop):
  rect = lambda A, B: np.array([A, (B[0], A[1]), B, (A[0], B[1])], np.float32)
  crop = np.array(crop, np.float32).reshape((2, 2))
  ptA, ptB = np.multiply(crop, image.shape[:2][::-1])
  size = np.abs(np.subtract(ptA, ptB))
  M = cv2.getPerspectiveTransform(rect(ptA, ptB), rect((0, 0), size))
  return (
    cv2.warpPerspective(image, M, tuple(size.astype(np.int))),
    np.array([
      x[0] for x in cv2.perspectiveTransform(np.array([ [x] for x in points ]), M)
    ], np.float32)
  )

class CFixedDataLoader(Sequence):
  def __init__(self, samples, preprocess, crops, gaussian):
    self._preprocess = preprocess
    self._samples = samples
    self._crops = crops
    self._gaussian = gaussian
    return

  def __len__(self):
    return len(self._samples)

  def on_epoch_end(self):
    return
  
  def __getitem__(self, index):
    X = []
    XConfidence = []
    YCoords = []
    YMasks = []
    
    image, points, confidence = self._samples.decode(self._samples.samples[index])
    points = np.array(points, np.float32)
    validPoints = np.array([(0 <= x) and (0 <= y) for x, y in points])
    
    for crop in self._crops:
      imageCrop, coords = _cropOf(image, points, crop)
      coords = np.divide(coords[:, ::-1], imageCrop.shape[:2])
      coords[~validPoints] = -1
      
      imageCrop = self._preprocess(imageCrop)
      X.append(imageCrop)
      XConfidence.append(confidence)
      YCoords.append(coords)
      
      HW = imageCrop.shape[0]
      masks = [self._gaussian((int(x * HW), int(y * HW))) for x, y in coords]
      YMasks.append(np.transpose(masks, (1, 2, 0)))
      continue

    return (
      [np.array(X, np.float32), np.array(XConfidence, np.float32)],
      [np.array(YCoords, np.float32), np.array(YMasks, np.float32)]
    )
