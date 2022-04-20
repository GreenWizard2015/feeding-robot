import Utils
import os
import numpy as np

import tensorflow as tf
from CAnchorsDetector import CAnchorsDetector
from CDataLoader import CDataLoader
from sampleAugmentations import sampleAugmentations, DEFAULT_AUGMENTATIONS
from CVisualizePredictionsCallback import CVisualizePredictionsCallback
from CDetectorTrainer import CDetectorTrainer
from CBetterInfoCallback import CBetterInfoCallback
from CEMapCallback import CEMapCallback

from scripts.common import loadDataset
from CDatasetSamples import CDatasetSamples

def train(
  modelName,
  trainer=CDetectorTrainer,
  BATCH_SIZE=8,
  AUGMENTATIONS_PER_SAMPLE=16,
  HW=224,
  EPOCHS=250,
  seed=42,
  optimizer=lambda: tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.),
  filepath=lambda *x: os.path.join(os.path.dirname(__file__), *x),
  gaussianRadius=5,
  trainLoaderArgs={},
  EMapArgs={},
  trainerArgs={}
):
  IMAGE_SIZE = (HW, HW)
  gaussian = Utils.gaussian(HW, gaussianRadius)
  if not(seed is None): Utils.resetSeed(seed)
  
  REAL_DATASET = loadDataset()
  trainingSamples, testValDS = Utils.splitDataset(REAL_DATASET, fraction=0.7)
  testingSamples, validationSamples = Utils.splitDataset(testValDS, fraction=0.5)
  
  print('Training dataset')
  Utils.datasetInfo(trainingSamples)
  print('\nValidation dataset')
  Utils.datasetInfo(validationSamples)
  print('\nTesting dataset')
  Utils.datasetInfo(testingSamples)
  print('..............')

  trainingSamples = CDatasetSamples(trainingSamples)
  testingSamples = CDatasetSamples(testingSamples)
  validationSamples = CDatasetSamples(validationSamples)
  #####
  CROPS = Utils.TEST_CROPS
  print('Crops per sample: %d' % (len(CROPS), ))
  
  valDataset = Utils.fixedDatasetFrom(
    validationSamples,
    lambda img: CAnchorsDetector.preprocessImage(img, IMAGE_SIZE),
    crops=CROPS,
    gaussian=gaussian
  )
  
  testDataset = Utils.fixedDatasetFrom(
    testingSamples,
    lambda img: CAnchorsDetector.preprocessImage(img, IMAGE_SIZE),
    crops=CROPS,
    gaussian=gaussian
  )
  
  # use one of samples for debug
  (debugSample, _), (debugSampleCoords, debugSampleMasks) = Utils.takeSample(valDataset)
  
  model = CAnchorsDetector(IMAGE_SIZE)
  trainer = trainer(detector=model.network, heatmapsWH=HW, **trainerArgs)
  trainer.compile(optimizer=optimizer(), loss=None)
  
  trainGenerator = CDataLoader(
    trainingSamples,
    {
      'batch size': BATCH_SIZE,
      'augmentations per sample': AUGMENTATIONS_PER_SAMPLE,
      'preprocess': model.preprocess,
      'gaussian': gaussian,
      'transformer': sampleAugmentations(
        **DEFAULT_AUGMENTATIONS,
        resize=IMAGE_SIZE,
      ),
      **trainLoaderArgs
    }
  )
  
  model_h5 = filepath('weights', 'trainer.h5')
  os.makedirs(os.path.dirname(model_h5), exist_ok=True)
  
  history = trainer.fit(
    trainGenerator,
    verbose=2,
    validation_data=valDataset,
    callbacks=[
      CVisualizePredictionsCallback(
        folder=filepath('debug', modelName),
        model=model.network,
        sample=debugSample, sampleCoords=debugSampleCoords, sampleMasks=debugSampleMasks
      ),
      tf.keras.callbacks.ModelCheckpoint(
        filepath=model_h5,
        monitor='val_loss', mode='min',
        verbose=1, save_best_only=True, save_weights_only=True
      ),
      CBetterInfoCallback(),
      CEMapCallback(trainer, trainGenerator, **EMapArgs)
    ],
    epochs=EPOCHS
  ).history
  # load best
  trainer.load_weights(model_h5)
  # save only model
  model.save(modelName, folder=filepath('weights'))
  
  evaluated = trainer.evaluate(testDataset)
  evaluatedStats = {m.name: value for m, value in zip(trainer.metrics, evaluated)}
  
  Utils.saveMetrics(history, lambda name: filepath('debug', modelName, 'metric_' + name))
  Utils.save(filepath('debug', modelName, 'stats.json'), trainGenerator.stats)
  Utils.save(filepath('debug', modelName, 'eval.json'), evaluatedStats)
  return model, trainer