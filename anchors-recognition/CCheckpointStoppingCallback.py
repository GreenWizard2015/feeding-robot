import tensorflow as tf
import os

class CCheckpointStoppingCallback(tf.keras.callbacks.Callback):
  def __init__(self, checkpoints):
    super().__init__()
    self._checkpoints = checkpoints
    return
  
  def on_epoch_end(self, epoch, logs=None):
    if not logs: return
    epoch = int(epoch) + 1 # 0 -> 1, 9 -> 10, etc.
    if epoch in self._checkpoints:
      data = self._checkpoints[epoch]
      for nm, minValue in data.items():
        v = float(logs[nm])
        if minValue < v:
          self.model.stop_training = True
    return