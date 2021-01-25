import tensorflow as tf
from scipy.ndimage.measurements import center_of_mass
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1 * 1024)]
)

from BasicDataGenerator import CBasicDataGenerator
from sampleAugmentations import sampleAugmentations
import cv2
import numpy as np
from CAnchorsDetector import CAnchorsDetector
from math import ceil

detector = CAnchorsDetector()
detector.load(kind='latest')

gen = CBasicDataGenerator({
  'folder': 'dataset/train',
  'batch size': 1,
  'batches per epoch': 1,
  'image size': (224, 224),
  'min anchors': 3,
  'target radius': 10,
  'transformer': sampleAugmentations(
    paddingScale=5.,
    rotateAngle=60.,
    brightnessFactor=.5,
    noiseRate=0.05
  ),
  'preprocess': detector.preprocess
})

IMAGE_PER_ROW = 4
while True:
  gen.on_epoch_end()
  X, Y = gen[0]
  for img, masks in zip(X, Y):
    images = [None] * IMAGE_PER_ROW
    images[0] = img
    
    pred = detector.network.predict(np.array([img]))[0]
    for i in range(8):
      mask = np.zeros((224, 224, 3))
      mask[:, :, 2] = masks[:, :, i]
      
      correctPos = center_of_mass(masks[:, :, i]) if np.any(0 < masks[:, :, i]) else np.array([0, 0])
      predPos = np.array([0, 0])
      if np.any(.05 < pred[:, :, i]):
        mask[:, :, 1] = pred[:, :, i] * 255.
        predPos = center_of_mass(pred[:, :, i])
        
        anchor = tuple(int(x) for x in predPos[::-1])
        color = (255, 0, 0)
        cv2.circle(mask, anchor, 8, color, 2)
        
      else:
        cv2.putText(mask, 'Low value', (20, 20), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 0, 0))
      
      coords = np.stack([correctPos, predPos])
      if np.all(0 <= coords) and np.all(coords <= 224):
        d = np.sqrt((np.subtract(correctPos, predPos) ** 2).sum())
        if 0 < d:
          cv2.putText(mask, 'error = %.1fpx' % d, (20, 210), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255))
      images.append(mask)
    
    rows = ceil(len(images) / IMAGE_PER_ROW)
    output = np.ones(((224 + 4) * rows, (224 + 4) * IMAGE_PER_ROW, 3)) * 255
    
    for i, img in enumerate(images):
      if not(img is None):
        row = i // IMAGE_PER_ROW
        col = i % IMAGE_PER_ROW
        X = 2 + (224 + 4) * col
        Y = 2 + (224 + 4) * row
        output[Y:Y+224, X:X+224] = img

    cv2.imshow('src', output)
    if 27 == cv2.waitKey(0): exit()
