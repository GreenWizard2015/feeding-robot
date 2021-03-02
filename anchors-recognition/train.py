import os
import sys
sys.path.append('../') # fix resolving in colab and eclipse ide

import tensorflow as tf
if 'COLAB_GPU' not in os.environ: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

import numpy as np
import detectorDiscriminator
import time

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from CAnchorsDetector import CAnchorsDetector
from BasicDataGenerator import CBasicDataGenerator
from sampleAugmentations import sampleAugmentations

def coordsGrid(values):
  xdim, ydim = K.int_shape(values)[1:-1]
  # Make array of coordinates
  ii, jj = tf.meshgrid(tf.range(xdim), tf.range(ydim), indexing='ij')
  coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,))], axis=-1)
  return tf.cast(coords, tf.float32)

def centerOfMass(values, coords):
  values = K.permute_dimensions(values, (0, 3, 1, 2))
  vdim = K.shape(values)[1]
  xdim = K.shape(values)[2]
  ydim = K.shape(values)[3]
  # Rearrange input into one vector per volume
  values_flat = tf.reshape(values, [-1, vdim, xdim * ydim, 1]) + K.epsilon()
  # Compute total mass for each volume
  total_mass = tf.reduce_sum(values_flat, axis=2)
  # Compute center of mass
  return tf.reduce_sum(values_flat * coords, axis=2) / total_mass

def euclideanDistLoss(ytrue, ypred, W):
  coords = coordsGrid(ypred)
  
  comA = centerOfMass(ytrue, coords)
  comB = centerOfMass(ypred, coords)
  
  distances = K.sqrt( tf.reduce_sum(K.square(comA - comB), axis=-1) ) * W
  # huber loss per class
  delta = 1.
  errors = distances
  condition = tf.less(errors, delta)
  small_res = 0.5 * tf.square(errors)
  large_res = delta * errors - 0.5 * tf.square(delta)
  return tf.where(condition, small_res, large_res)

def diceLoss(y_true, y_pred):
  axes = tuple(range(1, len(y_pred.shape) - 1)) # [batch dim] + [1, 2] + [classes] 
  numerator = 2. * K.sum(y_pred * y_true, axes)
  denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
  return 1 - K.mean(numerator / (denominator + K.epsilon()))

def complex_loss():
  W = [.1 for _ in range(9)] # 10px * .1 = 1.0
  W[-1] = 0. # remove euclidean loss for "none" class
  
  diceWeight = 1.0
  euclideanWeight = 1.0
  
  def calc(y_true, y_pred):
    dice = diceLoss(y_true, y_pred) * diceWeight
    euclidean = euclideanDistLoss(y_true, y_pred, W) * euclideanWeight
    return dice + euclidean
  
  return calc

###################
def advLoss(scale):
  def calc(_, discriminatorRate):
    return tf.reduce_mean(1. - discriminatorRate) * scale
  return calc

def flattenSamples(x):
  x = x[:, :, :, :-1]
  return K.reshape(
    K.permute_dimensions(x, (0, 3, 1, 2)),
    (-1, K.int_shape(x)[1], K.int_shape(x)[2], 1)
  )
  
def makeTrainer(mainModel, discriminator):
  mainInput = keras.layers.Input(shape=mainModel.input_shape[1:])
  rateScale = tf.Variable(0.0)
  
  predicted = mainModel(mainInput)
  discriminatorRate = discriminator(
    keras.layers.Lambda(flattenSamples)(predicted)
  )

  model = keras.Model(inputs=[mainInput], outputs=[predicted, discriminatorRate])
  model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4, clipnorm=1.),
    loss=[complex_loss(), advLoss(rateScale)]
  )
  return model, rateScale

###################
discriminator = detectorDiscriminator.CDetectorDiscriminator(
  size=(224, 224),
  samplesPreprocessor=lambda x: K.eval(flattenSamples(x)),
  historySize=1024,
  probBase=2 # probabilities: 0 = x, 1 = 1/2, 2 = 1/4, ..., up to 1/2^BATCH_SIZE
)
discriminator.network.summary()
#####
BATCH_SIZE = 16
TRAIN_EPOCHS = 32
DISCRIMINATOR_EPOCHS = 32

model = CAnchorsDetector()
model.network.summary()

trainer, advLossPower = makeTrainer(model.network, discriminator.network)
trainer.summary()

folder = lambda x: os.path.join(os.path.dirname(__file__), x)

COMMON_GENERATOR_SETTINGS = {
  'min anchors': 0,
  'batch size': BATCH_SIZE,
  'image size': (224, 224),
  'target radius': 10,
  'fake output': True,
  'preprocess': model.preprocess,
}

trainGenerator = CBasicDataGenerator({
  **COMMON_GENERATOR_SETTINGS,
  'folder': folder('dataset/train'),
  'batches per epoch': TRAIN_EPOCHS,
  'transformer': sampleAugmentations(
    paddingScale=5.,
    rotateAngle=120.,
    brightnessFactor=.5,
    noiseRate=0.05,
    resize=(224, 224)
  ),
})

discriminatorGenerator = CBasicDataGenerator({
  **COMMON_GENERATOR_SETTINGS,
  'folder': folder('dataset/validation'),
  'batches per epoch': DISCRIMINATOR_EPOCHS,
  'transformer': sampleAugmentations(
    paddingScale=3.,
    rotateAngle=50.,
    brightnessFactor=.5,
    noiseRate=0.01,
    resize=(224, 224)
  ),
})

# create folder for weights
os.makedirs(os.path.dirname(model.weights_file('')), exist_ok=True)

totalEpochs = 0
lastMaxLoss = 0.0
for i in range(1, 100):
  print('Super epoch %d.' % i)
  
  model.network.trainable = False
  DAcc = discriminator.train(
    discriminatorGenerator, model.network,
    epochs=50, minEpochs=5, minAcc=.95
  )
  model.network.trainable = True

  advLossPower.assign(lastMaxLoss * 0.5 * DAcc)
  history = trainer.fit(
    x=trainGenerator,
    verbose=2,
    callbacks=[
      keras.callbacks.EarlyStopping(monitor='D_loss', mode='min', patience=10)
    ],
    epochs=5 * i
  ).history
  
  lastMaxLoss = np.max(history['ADet_loss'])
  totalEpochs += len(history['loss'])
  model.network.save_weights(model.weights_file('%d' % totalEpochs))
