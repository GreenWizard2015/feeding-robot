import tensorflow as tf
import os
import time

class CEMapCallback(tf.keras.callbacks.Callback):
  def __init__(self, trainer, generator, period=0, scaleFactor=16, temperature=1, batchSize=16):
    super().__init__()
    self._trainer = trainer
    self._generator = generator
    self._scaleFactor = scaleFactor
    self._period = period
    self._temperature = temperature if callable(temperature) else lambda x: temperature
    self._batchSize = batchSize
    return
  
  def on_epoch_begin(self, epoch, logs=None):
    if self._period < 1: return
    if 0 < (epoch % self._period): return
    
    print('Mining error maps...  ', end='')
    T = time.time()
    temperature = self._temperature(epoch)
    self._generator.updateEMap(
      lambda a, b: self._trainer.evaluateErrorMap(a, b, self._scaleFactor, temperature).numpy(),
      self._batchSize
    )
    print('%.1f sec' % (time.time() - T))
    return