import tensorflow as tf
import os

class CBetterInfoCallback(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    return
  
  def on_epoch_end(self, epoch, logs=None):
    if not logs: return
    category = ['Training', 'Testing']
    res = {x: {} for x in category}
    for name, value in logs.items():
      C = category[int(name.startswith('val_'))]
      res[C][name] = float(value)
      continue
    
    for title, data in res.items():
      print(title)
      print('=' * len(title))
      for k, v in data.items():
        print('%s: %.6f' % (k, v))
      print('', flush=True)
    return