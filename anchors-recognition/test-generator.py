#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setupEnv()

from CDataLoader import CDataLoader
from sampleAugmentations import sampleAugmentations, DEFAULT_AUGMENTATIONS
import cv2
import numpy as np
from CAnchorsDetector import CAnchorsDetector
from VisualizePredictions import VisualizePredictions
from CDatasetSamples import CDatasetSamples

detector = CAnchorsDetector()
detector.network.summary()

detector.load(kind='xxx', folder='d:/')
# detector.load(kind='focal', folder='weights')

gen = CDataLoader(
  CDatasetSamples(),
  {
  'batch size': 16,
  'batches per epoch': 1,
  'transformer': sampleAugmentations(
    **DEFAULT_AUGMENTATIONS,
    resize=(224, 224),
  ),
  'preprocess': detector.preprocess,
  'gaussian': Utils.gaussian(224, 5),
  'prioritized sampling': True
})

while True:
  gen.on_epoch_end()
  [X, _], [YPos, YMasks] = gen[0]
  for img, Yposes, Ymasks in zip(X, YPos, YMasks):
    cv2.imshow('src', VisualizePredictions(detector.network, img, Yposes, Ymasks, zoomTarget=not True))
    if 27 == cv2.waitKey(0): exit()