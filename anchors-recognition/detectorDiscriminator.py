import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import numpy as np
import time

def soft_labels_acc(maxDelta):
  def f(y_true, y_pred):
    return K.mean(
      K.cast(K.abs(y_true - y_pred) < maxDelta, tf.float16),
      axis=-1
    )

  f.__name__ = 'soft acc'
  return f

LEAKY_RELU = {'activation': keras.layers.LeakyReLU(alpha=0.2)}
def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", **LEAKY_RELU)(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def downsamplingBlock(res, sz, filters):
  for _ in range(3):
    res = convBlock(res, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same", **LEAKY_RELU)(res)
  return res

def anchorsDetectorDiscriminatorModel(size, labelsMargin):
  res = inputs = layers.Input(shape=(*size, 1))

  res = downsamplingBlock(res, 7, 6)
  res = downsamplingBlock(res, 5, 6)
  res = downsamplingBlock(res, 5, 4)
  res = downsamplingBlock(res, 3, 4)
  res = downsamplingBlock(res, 3, 3)
  
  res = layers.Flatten()(res)
  res = layers.Dense(256, **LEAKY_RELU)(res)
  res = layers.Dense(128, **LEAKY_RELU)(res)
  res = layers.Dense(32, **LEAKY_RELU)(res)
  
  model = keras.Model(
    inputs=inputs,
    outputs=layers.Dense(1, activation='sigmoid')(res),
    name='D'
  )
  model.compile(
    loss=keras.losses.Huber(delta=1.0), # because we use soft labels
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[soft_labels_acc(labelsMargin / 2)]
  )
  return model

class CDetectorDiscriminator:
  def __init__(self, size, samplesPreprocessor, labelsMargin=0.1, historySize=1024, probBase=np.e):
    self._labelsMargin = labelsMargin
    self.network = anchorsDetectorDiscriminatorModel(size, labelsMargin)
    self.network.trainable = False
    self._samplesPreprocessor = samplesPreprocessor
    self._history = np.random.random_sample((historySize, *size, 1))
    self._probBase = float(probBase)
    return
  
  def fitLabel(self, X, isReal):
    labelShift = 1. -  self._labelsMargin if isReal else 0.0
    Y = labelShift + np.random.random_sample((X.shape[0], )) * self._labelsMargin
    return self.network.fit(X, Y, verbose=0).history
  
  def _makeFakeBatch(self, fakes):
    rng = np.arange(fakes.shape[0])
    prob = np.power(self._probBase, -rng)
    replaceByOldN, saveN = np.random.choice(rng, p=prob / np.sum(prob), replace=True, size=(2,))
    
    if 0 < saveN:
      N = saveN
      indH = np.random.choice(np.arange(self._history.shape[0]), replace=False, size=(N,))
      indF = np.random.choice(np.arange(fakes.shape[0]), replace=False, size=(N,))
      self._history[indH] = fakes[indF]
      
    if 0 < replaceByOldN:
      N = replaceByOldN
      indH = np.random.choice(np.arange(self._history.shape[0]), replace=False, size=(N,))
      indF = np.random.choice(np.arange(fakes.shape[0]), replace=False, size=(N,))
      fakes[indF] = self._history[indH]
       
    return fakes
  
  def train(
    self, generator, predictor,
    epochs=100, minAcc=0.95, minEpochs=10
  ):
    self.network.trainable = True
    
    for epoch in range(epochs):
      T = time.time()
      losses = []
      acc = []
      def fit(X, isReal):
        res = self.fitLabel(X, isReal)
        losses.append(res['loss'][0])
        acc.append(res['soft acc'][0])
        return
      
      generator.on_epoch_end()
      for batchID in range(len(generator)):
        samples, (trueX, _) = generator[batchID]
        # train real
        fit(self._samplesPreprocessor(trueX), True)
        # train on generated fakes
        fakes = self._makeFakeBatch( self._samplesPreprocessor(predictor(samples)) )
        fit(fakes, False)

      #
      avgAcc = np.mean(acc)
      avgLoss = np.mean(losses)
      print(
        'Epoch %d. Time: %.1f sec. Avg. loss: %.4f. Avg. acc: %.2f.' % (
          epoch, time.time() - T, avgLoss, avgAcc
        )
      )
      if (minEpochs <= epoch) and (minAcc <= avgAcc): break

    self.network.trainable = False
    return avgAcc # latest accuracy
