#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setupEnv()
Utils.resetSeed(42)

from CAnchorsDetector import CAnchorsDetector, CStackedDetector
import numpy as np
from scripts.common import loadDataset
from CDatasetSamples import CDatasetSamples
import os
import json

def evaluateModel(detector, saveTo, dataset):
  methodsNames = ['simple', 'full', 'all-avg', 'all-prod', 'all-min', 'all-max']
  methodsEval = [{} for _ in methodsNames]
  
  for ind, sample in enumerate(dataset.samples):
    print('process sample %d' % (ind, ))
    image, points, _ = dataset.decode(sample)
    points = np.array(points, np.float32)[:, ::-1]
  
    res = [
      detector.detect(image, raw=True),
      detector.combinedDetections(image, raw=True, crops='full'),
    ]
    res.extend(detector.combinedDetections(
      image, raw=True,
      crops='all', mode=['avg', 'prod', 'min', 'max']
    ))
    
    for methodMetrics, raw in zip(methodsEval, res):
      errorPerPoint = {}
      for i, (realPt, (predPos, predProb)) in enumerate(zip(points, raw)):
        predPos = np.array(predPos)
        visible = np.all(0 <= realPt)
        errorPerPoint[i] = {
          'visible': int(visible),
          'prob': float(predProb),
          'dist': float(np.linalg.norm(realPt - predPos) if visible else -1)
        }
        continue
      methodMetrics[ind] = errorPerPoint
    continue
  
  with open(saveTo('eval.json'), 'w') as f:
    json.dump(
      {name: data for name, data in zip(methodsNames, methodsEval)},
      f, indent=2
    )
  return

FOLDER = 'evaluated-metrics'
os.makedirs(FOLDER, exist_ok=True)

excludeTrainSamples=True
REAL_DATASET = dataset = loadDataset()
if excludeTrainSamples:
  Utils.resetSeed(42)
  _, dataset = Utils.splitDataset(REAL_DATASET, fraction=0.7)
dataset = CDatasetSamples(dataset)

for model_h5 in CAnchorsDetector.models('weights'):
  modelName = os.path.basename(model_h5)[:-3].replace('anchors-detector-', '')
  print(modelName)
  evaluateModel(
    CAnchorsDetector().load(file=model_h5),
    saveTo=lambda x: os.path.join(FOLDER, '%s-%s' % (modelName, x)),
    dataset=dataset
  )
  continue