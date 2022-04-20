import tensorflow as tf
import os
import cv2
from VisualizePredictions import VisualizePredictions

class CVisualizePredictionsCallback(tf.keras.callbacks.Callback):
  def __init__(self, folder, model, sample, sampleCoords, sampleMasks):
    super().__init__()
    
    self._folder = folder
    self._params = [model, sample, sampleCoords, sampleMasks]
    return
  
  def on_epoch_end(self, epoch, logs=None):
    img = VisualizePredictions(*self._params)
    file = os.path.join(self._folder, '%06d.png' % (epoch + 1,))
    os.makedirs(self._folder, exist_ok=True)
    cv2.imwrite(file, img)
    return