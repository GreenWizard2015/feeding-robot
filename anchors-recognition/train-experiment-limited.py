#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setupEnv()

import os
import numpy as np
import math
import random
import time
import json
from collections import defaultdict

import tensorflow as tf
from CAnchorsDetector import CAnchorsDetector, CStackedDetector
from CDataLoader import CDataLoader
from sampleAugmentations import sampleAugmentations, DEFAULT_AUGMENTATIONS
from CVisualizePredictionsCallback import CVisualizePredictionsCallback
from CBetterInfoCallback import CBetterInfoCallback
from CCheckpointStoppingCallback import CCheckpointStoppingCallback
from CDetectorTrainer import CDetectorTrainer
import NNUtils

from scripts.common import loadDataset
from CDatasetSamples import CDatasetSamples

MIN_ACCURACY = 0.90
ACCURACY_METRIC = 'accuracy_adaptive_vis'

USE_SAME_SAMPLES = False

FIXED_SAMPLES = True
SKIP_FRAMES = 15

TTA = {'mode': 'flip', 'penalize': True}
DETECTION_MINE_ARGS = {'crops': 'all', 'mode': 'prod', 'TTA': TTA}
DETECTION_SCORE_ARGS = {'crops': 'full', 'mode': 'prod', 'TTA': TTA}

DROP_SAMPLES = False
CONFIDENCE_THRESHOLD = 0.9

MIN_SAMPLES_CLUSTERS = 15

INIT_BEST_MODEL = True
ENSEMBLE_MODELS_N = 1
ENSEMBLE_MODELS_POOL = 1

STACK_ENSEMBLES = False

TOTAL_ROUNDS = 100

GAUSSIAN_RADIUS = 1 # px
MASK_LOSS = NNUtils.focalLoss2d()

BATCH_SIZE = 8
SAMPLES_PER_EPOCH = BATCH_SIZE * 126
HW = 224
IMAGE_SIZE = (HW, HW)
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 20
gaussian = Utils.gaussian(HW, GAUSSIAN_RADIUS)

def filepath(*x):
  res = os.path.join(os.path.dirname(__file__), *x)
  os.makedirs(os.path.dirname(res), exist_ok=True)
  return res

def trainGenerator(samples):
  return CDataLoader(
    samples,
    {
      'batch size': BATCH_SIZE,
      'samples per epoch': SAMPLES_PER_EPOCH,
      'preprocess': lambda img: CAnchorsDetector.preprocessImage(img, IMAGE_SIZE),
      'min anchors': (0, 8),
      'gaussian': gaussian,
      'transformer': sampleAugmentations(
        **DEFAULT_AUGMENTATIONS,
        resize=IMAGE_SIZE,
      ),
      'prioritized sampling': True
    }
  )

def valGenerator(samples):
  return Utils.fixedDatasetFrom(
    samples,
    lambda img: CAnchorsDetector.preprocessImage(img, IMAGE_SIZE),
    crops=Utils.TEST_CROPS,
    gaussian=gaussian
  )

def fitModel(trainSamples, valSamples, modelName, initBestModel):
  model = CAnchorsDetector(IMAGE_SIZE)
  if initBestModel:
    model.load('best-limited', folder=filepath('weights'))
    # forget last layer
    L = model.network.layers[-2]
    L.set_weights([np.random.normal(scale=1e-2, size=x.shape) for x in L.get_weights()])

  trainer = CDetectorTrainer(
    detector=model.network,
    masksLoss=MASK_LOSS,
    weights={'masks': 1.0},
    useAugmentations=True,
    argmaxAccuracy=True,
    adaptiveAccuracyD=5,
  )
  trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.), loss=None)
  
  print('Training samples: %d' % (len(trainSamples), ))
  print('Validation samples: %d' % (len(valSamples), ))
  
  valDataset = valGenerator(valSamples)
  # use one of val. sample for debug
  (debugSample, _), (debugSampleCoords, debugSampleMasks) = Utils.takeSample(valDataset)
  
  history = trainer.fit(
    trainGenerator(trainSamples),
    verbose=2,
    validation_data=valDataset,
    callbacks=[
      CVisualizePredictionsCallback(
        folder=filepath('debug', modelName),
        model=model.network,
        sample=debugSample, sampleCoords=debugSampleCoords, sampleMasks=debugSampleMasks
      ),
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
      ),
      CBetterInfoCallback()
    ],
    epochs=1000000 # just big number
  ).history

  Utils.saveMetrics(history, lambda name: filepath('debug', modelName, 'metric_' + name))
  return model, trainer

def trainModel(trainSamples, valSamples, modelName, initBestModel):
  print('Start training "%s".' % (modelName,))
  model, trainer = fitModel(trainSamples, valSamples, modelName, initBestModel)
  model.save(modelName, folder=filepath('weights'))
  print('Model "%s" trained.' % (modelName,))
  return model, trainer

def estimateThreshold(detector, size=(720, 1280, 3), N=64, percentile=95):
  maxTh = [] # N, 8
  for _ in range(N):
    _, heatmaps = detector.combinedDetections(
      (np.random.random(size) * 255).astype(np.uint8),
      returnHeatmaps=True, **DETECTION_MINE_ARGS
    )
    
    maxTh.append([x.max() for x in heatmaps])
    continue
  
  return np.percentile(maxTh, percentile, axis=0)

def mineSamples(models):
  print('Mine samples....')
#   threshold = estimateThreshold(CStackedDetector(models, cache=None))
#   print('Estimated thresholds: ', threshold)

  detector = CStackedDetector(
    models,
    cache='cache' if STACK_ENSEMBLES else None
  )
  # samples mining
  newSamples = Utils.extractSamples(
    lambda frame: detector.combinedDetections(frame, **DETECTION_MINE_ARGS),
    # skip some frames
    SKIP_AFTER_DETECTION=SKIP_FRAMES,
    minFrame=random.randint(0, SKIP_FRAMES - 1) if not FIXED_SAMPLES else 0
  )
#   newSamples = Utils.dropPoints(newSamples, threshold=threshold)
  newSamples = Utils.normalizeConfidence(newSamples)
  Utils.save(filepath('debug', 'mined-samples-%.0f.json' % (time.time())), Utils.samples2dataset(newSamples))
  print('Collected %d samples' % len(newSamples))

#   filtered = []
#   for A, B, points in newSamples:
#     save = any((CONFIDENCE_THRESHOLD < x.get('confidence', 1)) for x in points.values())
#     if save:
#       filtered.append((A, B, points))
  samples = newSamples
  print('Filtered %d samples' % len(samples))
  return samples

def ss2dataset(samples):
  return Utils.samples2dataset([(x[0], x[1], x[2]) for x in samples])

def choiceSamples(samplesClusters, N, usedSamples=None, distribution='test'):
  usedSamples = set() if usedSamples is None else usedSamples
  totalSamples = sum([len(x) for x in samplesClusters])
  assert N <= totalSamples
  
  def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
  
  clustersI = np.arange(len(samplesClusters))
  clustersP = None
  if 'test' == distribution:
    mainP = gaussian(clustersI, clustersI[0], clustersI[-1] / 2.0)
    m = len(samplesClusters) * (1.0 / 3.0)
    secondaryP = gaussian(clustersI, m, m / 2.0)
    clustersP = mainP + secondaryP
    
  if 'train' == distribution:
    mainP = gaussian(clustersI, clustersI[0], clustersI[-1] / 2.0)
    m = len(samplesClusters) * (1.0 / 3.0)
    secondaryP = gaussian(clustersI, m * 2.0, m / 2.0)
    clustersP = mainP + secondaryP
  assert not(clustersP is None)
  res = []
  while (len(res) < N) and (len(usedSamples) < totalSamples):
    ind = random.choices(clustersI, weights=clustersP)[0]  
    samples = [x for x in samplesClusters[ind] if not ((x[0], x[1]) in usedSamples)]
    if samples:
      sample = random.choice(samples)
      ID = (sample[0], sample[1])
      if not(ID in usedSamples):
        usedSamples.add(ID)
        res.append(sample)
    else:
      clustersP[ind] = 0
    continue
  return res, usedSamples

def samples2clusters(samples):
  if len(samples) < 2: return [samples]

  errors = np.array([x[-1] for x in samples]).reshape((-1, 1))
  thresh = 0.1 * (errors.max() - errors.min())

  import scipy.cluster.hierarchy as hcluster
  from collections import Counter
  clustersID = []
  while True:
    clustersID = hcluster.fclusterdata(errors, thresh, criterion="distance")
    cnt = Counter(clustersID)
    ((_, biggestCluster),) = cnt.most_common(1)
    if (biggestCluster <= 1) or (MIN_SAMPLES_CLUSTERS < len(cnt)): break
    thresh *= 0.75
  
  clusters = defaultdict(list)
  for cid, sample in zip(clustersID, samples):
    clusters[cid].append(sample)
    continue
  
  clusters = list(clusters.values())
  clusters = list(sorted(clusters, key=lambda x: x[-1][-1]))
  return clusters

def trainEnsemble(samples, valDataset, ensembleName, N, M, initBestModel):
  models = []
  samplesN = len(samples)
  clusteredSamples = samples2clusters(samples)
  selectedSamplesN = min((len(clusteredSamples) * 2, samplesN // 2))
  
  Utils.save(filepath('debug', '%s-clusters.json' % (ensembleName, )), clusteredSamples)
  while len(models) < N:
    modelName = '%s-%d' % (ensembleName, len(models))
    ############
    validationSamples = trainingSamples = samples
    if 1 < len(samples):
      trainingSamples, usedSamples = choiceSamples(
        clusteredSamples, selectedSamplesN,
        distribution='train'
      )
      validationSamples = trainingSamples
      if not USE_SAME_SAMPLES:
        validationSamples, _ = choiceSamples(
          clusteredSamples, selectedSamplesN,
          usedSamples,
          distribution='test'
        )
        if len(validationSamples) < 1: # use best samples
          validationSamples = clusteredSamples[0]
      ##########
  
    Utils.save(filepath('debug', '%s-train.json' % (ensembleName, )), ss2dataset(trainingSamples))
    Utils.save(filepath('debug', '%s-val.json' % (ensembleName, )), ss2dataset(validationSamples))
    ############

    model, trainer  = trainModel(
      CDatasetSamples(ss2dataset(trainingSamples)),
      CDatasetSamples(ss2dataset(validationSamples)),
      modelName,
      initBestModel=initBestModel
    )
    # evaluate on "real" samples
    stats = trainer.evaluate(valDataset, return_dict=True)
    modelLoss = stats['loss']
    modelAcc = stats[ACCURACY_METRIC]
    if modelAcc < MIN_ACCURACY:
      print('Model was rejected due to low accuracy. (%.3f)' % (modelAcc,))
      continue
    models.append((model, modelLoss))
    print('Model was accepted. Loss: %.4f, accuracy: %.3f.' % (modelLoss, modelAcc))
    print('Models: ' + ', '.join([('%.4f' % loss) for _, loss in models]))
    print('-----------------------------------------------')
    continue
  
  models = list(sorted(models, key=lambda x: x[-1]))
  models[0][0].save('best-limited', folder=filepath('weights'))
  return [model for model, _ in models[:M]]

def scoredSamples(samples, ensemble):
  detector = CStackedDetector(ensemble, cache=None)
  samples = CDatasetSamples(samples=[(x[0], x[1], x[2]) for x in samples])
  res = []
  E = []
  for sample in samples.samples:
    img, points, _ = samples.decode(sample)
    points = np.array(points, np.float32)[:, ::-1]
    raw = sample['raw']
    preds = detector.combinedDetections(img, **DETECTION_SCORE_ARGS, raw=True)
    error = 0.0
    errors = [0.0] * 8
    for i, (realPt, (predPos, predProb)) in enumerate(zip(points, preds)):
      predPos = np.array(predPos)
      visible = np.all(0 <= realPt)
      dist = float(np.linalg.norm(realPt - predPos) if visible else 0)
      errors[i] = dist 
      error += min((2.0, dist / 15.0 ))#px
      continue
    
    E.append({
      'file': raw[0],
      'frame': raw[1],
      'errors': [float(x) for x in errors],
      'confidence': [float(x.get('confidence', 1)) for x in raw[2].values()]
    })
    res.append((*raw, error))
    continue
  
  Utils.save(filepath('debug', 'errors.json'), E)
  exit()
  return res

def combineSamples(*mergingSamples):
  indices = {}
  res = []
  for samples in mergingSamples:
    for sample in samples:
      ID = (sample[0], sample[1])
      if ID in indices:
        res[indices[ID]] = sample
      else: # replace same samples
        indices[ID] = len(res)
        res.append(sample)
      continue
    continue
  return res
  
def main():
  Utils.resetSeed(42)
  REAL_DATASET = loadDataset()
  
  # take one samples for training
  #trainingSamples = [random.choice(Utils.flattenDataset(REAL_DATASET))]
  trainingSamples = [Utils.flattenDataset(REAL_DATASET)[0]]

  tmp = CDatasetSamples(Utils.samples2dataset(trainingSamples))
  _, pts, _ = tmp.decode(tmp.samples[0])
  assert np.all(0 <= np.array(pts)), 'All points must be visible. Bad seed.'

  realValDataset = valGenerator(CDatasetSamples(Utils.samples2dataset(trainingSamples)))
  
  trainingSamples = [[*x, 1.0] for x in trainingSamples]
  # train basic ensemble
  ensemble = trainEnsemble(
    trainingSamples, realValDataset,
    ensembleName='limited-basic',
    N=1, M=1,
    initBestModel=True # False
  )
  
  fakeDataset = []
  for currentRound in range(1, TOTAL_ROUNDS):
    newSamples = mineSamples(ensemble)
    fakeDataset = newSamples if DROP_SAMPLES else combineSamples(fakeDataset, newSamples)
    fakeDataset = scoredSamples(fakeDataset, ensemble)
    Utils.save(filepath('debug', 'scored-samples-e%06d.json' % (currentRound, )), fakeDataset)
    Utils.save(filepath('debug', 'samples-e%06d.json' % (currentRound, )), ss2dataset(fakeDataset))

    newEnsemble = trainEnsemble(
      fakeDataset, realValDataset,
      ensembleName='limited-%d' % currentRound,
      N=ENSEMBLE_MODELS_N, M=ENSEMBLE_MODELS_POOL,
      initBestModel=INIT_BEST_MODEL
    )
    
    if STACK_ENSEMBLES:
      ensemble.extend(newEnsemble)
    else:
      ensemble = newEnsemble
    continue
  return

if __name__ == "__main__":
  main()
