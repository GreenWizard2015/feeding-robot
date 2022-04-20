import tensorflow as tf
import tensorflow.keras.backend as K
import NNUtils

class HeatmapDecodingLayer(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    return

  def _preprocess(self, values):
    values = NNUtils.softmax2d(values, False)
    values = tf.math.pow(values, 8.0)
    values = NNUtils.norm2d(values)
    values = tf.math.pow(values, 8.0)
    values = NNUtils.norm2d(values)
    values = tf.math.pow(values, 8.0)
    values = NNUtils.norm2d(values)
    return values
  
  def call(self, data, scores=None):
    values = NNUtils.channelwise(data) # NHWC -> NCHW
    scores = values if scores is None else scores

    values = self._preprocess(values)
    pos = NNUtils.centerOfMassCHW(values)
    cellValue = NNUtils.valueReaderCHW(scores)
    return tf.concat([pos, cellValue(pos)[:, :, 0]], axis=-1)
