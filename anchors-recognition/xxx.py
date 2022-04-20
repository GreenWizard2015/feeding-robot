#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
import cv2
Utils.setupEnv()
Utils.resetSeed(42)

from CAnchorsDetector import CAnchorsDetector
import numpy as np
from scripts.common import loadDataset
from CDatasetSamples import CDatasetSamples
import os

def evaluateModel(detector, saveTo, dataset):
  methodsNames = ['simple', 'full', 'all-avg', 'all-prod', 'all-min']
  methodsEval = [{} for _ in methodsNames]
  
  for ind, sample in enumerate(dataset.samples):
    image, points, _ = dataset.decode(sample)
    cv2.imshow('img', image)
    cv2.waitKey()
    
    pred = detector.combinedDetections(
      image,
      raw=True, crops='all', mode=['uncertainty', 'prod', 'min', 'entropy'], returnHeatmaps=True
    )
    uncHeatmaps, prodHeatmaps, mH, eH = [x[-1] for x in pred]
    for name, heatmaps in [('unc', uncHeatmaps), ('prod', prodHeatmaps), ('min', mH), ('entropy', eH)]:
      for j, hm in enumerate(heatmaps):
        print(name, hm.min(), hm.max())
        hm = (hm / hm.max()) * 255.
        x, y = np.unravel_index(hm.argmax(), hm.shape)
        
        hm = cv2.applyColorMap(hm.astype(np.uint8), cv2.COLORMAP_JET)
        
        hm = cv2.addWeighted(image, 0.5, hm, 0.5, 0)
        cv2.circle(hm, (y, x), 7, (0, 0, 255), 3)
        cv2.imwrite('d:/xxx/%d_%s_%d.jpg' % (ind, name, j), hm)
        
    for j, (uncH, prodH) in enumerate(zip(uncHeatmaps, mH)):
#       uncH /= uncH.max()
      th = np.percentile(uncH, 99)
#       prodH /= prodH.max()
      hm = prodH# * (1 - uncH)
      hm[uncH > th] = 0
      hm = (hm / hm.max())* 255
      x, y = np.unravel_index(hm.argmax(), hm.shape)
      
      hm = cv2.applyColorMap(hm.astype(np.uint8), cv2.COLORMAP_JET)
      
      hm = cv2.addWeighted(image, 0.5, hm, 0.5, 0)
      cv2.circle(hm, (y, x), 7, (0, 0, 255), 3)
      cv2.imwrite('d:/xxx/%d_%s_%d.jpg' % (ind, 'comb', j), hm)
    break
    continue
  return

dataset = loadDataset()
dataset = CDatasetSamples(dataset)

for model_h5 in CAnchorsDetector.models('weights'):
  modelName = os.path.basename(model_h5)[:-3].replace('anchors-detector-', '')
  print(modelName)
  evaluateModel(
    CAnchorsDetector().load(file=model_h5),
    saveTo=lambda x: os.path.join(FOLDER, '%s-%s' % (modelName, x)),
    dataset=dataset
  )
  break
  continue