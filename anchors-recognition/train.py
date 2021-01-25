import os
import sys
sys.path.append('../') # fix resolving in colab and eclipse ide

import tensorflow as tf
if 'COLAB_GPU' not in os.environ: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

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

BATCH_SIZE = 16
TRAIN_EPOCHS = 32
TEST_EPOCHS = 8

model = CAnchorsDetector()
model.network.compile(
  optimizer=keras.optimizers.Adam(lr=0.001, clipnorm=1.),
  loss=complex_loss(),
  metrics=[]
)
model.network.summary()
folder = lambda x: os.path.join(os.path.dirname(__file__), x)

trainGenerator = CBasicDataGenerator({
  'folder': folder('dataset/train'),
  'batch size': BATCH_SIZE,
  'batches per epoch': TRAIN_EPOCHS,
  'image size': (224, 224),
  'min anchors': 2,
  'target radius': 10,
  'transformer': sampleAugmentations(
    paddingScale=5.,
    rotateAngle=120.,
    brightnessFactor=.5,
    noiseRate=0.05
  ),
  'preprocess': model.preprocess
})

validGenerator = CBasicDataGenerator({
  'folder': folder('dataset/validation'),
  'batch size': BATCH_SIZE,
  'batches per epoch': TEST_EPOCHS,
  'image size': (224, 224),
  'min anchors': 1,
  'target radius': 10,
  'transformer': sampleAugmentations(
    paddingScale=3.,
    rotateAngle=50.,
    brightnessFactor=.5,
    noiseRate=0.01
  ),
  'preprocess': model.preprocess
})

# create folder for weights
os.makedirs(
  os.path.dirname(model.weights_file()),
  exist_ok=True
)

model.network.fit(
  x=trainGenerator,
  validation_data=validGenerator,
  verbose=2,
  callbacks=[
    keras.callbacks.EarlyStopping(
      monitor='val_loss', mode='min',
      patience=500
    ),
    # best by validation loss
    keras.callbacks.ModelCheckpoint(
      filepath=model.weights_file('best'),
      save_weights_only=True,
      save_best_only=True,
      monitor='val_loss',
      mode='min',
      verbose=1
    ),
    # best by train loss
    keras.callbacks.ModelCheckpoint(
      filepath=model.weights_file('latest'),
      save_weights_only=True,
      save_best_only=True,
      monitor='loss',
      mode='min',
      verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss', factor=0.9,
      patience=50
    )
  ],
  epochs=1000000 # we use EarlyStopping, so just a big number
)