#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setupEnv()

import SimpleTrainingSetup
import CDetectorTrainer
import NNUtils

for D in [5, 10, 20]:
  SimpleTrainingSetup.train(
    modelName='coords-d%d' % D,
    trainer=lambda *args, **kwargs: CDetectorTrainer.CDetectorTrainer(
      *args, **kwargs,
      cellRadiusPx=D,
      weights={'coords': 1.0, 'probs': 1.0}
    )
  )
print('Done')
