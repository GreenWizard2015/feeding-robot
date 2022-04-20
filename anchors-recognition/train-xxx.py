#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setupEnv()

import SimpleTrainingSetup
import CDetectorTrainer
import NNUtils

SimpleTrainingSetup.train(
  modelName='xxx',
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.focalLoss2d(),
    weights={'masks': 1.0},
    useAugmentations=True,
  ),
  gaussianRadius=1,
  trainLoaderArgs={'prioritized sampling': True}
)
print('Done')
