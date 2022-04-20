#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setupEnv()

import SimpleTrainingSetup
import CDetectorTrainer
import NNUtils

SimpleTrainingSetup.train(
  modelName='focal-5px',
  gaussianRadius=5,
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.focalLoss2d(),
    weights={'masks': 1.0}
  )
)

SimpleTrainingSetup.train(
  modelName='dice-5px',
  gaussianRadius=5,
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.diceLoss,
    weights={'masks': 1.0}
  )
)

SimpleTrainingSetup.train(
  modelName='lcosh-dice-5px',
  gaussianRadius=5,
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.log_cosh_dice_loss,
    weights={'masks': 1.0}
  )
)

######################

SimpleTrainingSetup.train(
  modelName='focal-1px',
  gaussianRadius=1,
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.focalLoss2d(),
    weights={'masks': 1.0}
  )
)

SimpleTrainingSetup.train(
  modelName='dice-1px',
  gaussianRadius=1,
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.diceLoss,
    weights={'masks': 1.0}
  )
)

SimpleTrainingSetup.train(
  modelName='lcosh-dice-1px',
  gaussianRadius=1,
  trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
    *args, **kwargs,
    masksLoss=NNUtils.log_cosh_dice_loss,
    weights={'masks': 1.0}
  )
)

print('Done')
