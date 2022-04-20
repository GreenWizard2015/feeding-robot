#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob

FOLDER = 'evaluated-metrics'
METHODS = ['simple', 'full', 'all-avg', 'all-prod', 'all-min', 'all-max']
TITLES = {a: b for a, b in (zip(
  METHODS,
  [
    'One-shot, avg.',
    'One-shot, max.',
    'Multiple overlapped predictions (avg)',
    'Multiple overlapped predictions (prod)',
    'Multiple overlapped predictions (min)',
    'Multiple overlapped predictions (max)',
  ]
))}
###################
MODELS = {}
for name in glob.iglob(os.path.join(FOLDER, '*.json')):
  with open(name, 'r') as f:
    modelName = os.path.basename(name).replace('-eval.json', '')
    MODELS[modelName] = json.load(f)
  continue
print(MODELS.keys())
###################
def modelDistErrors(data, maxDist=50):
  plt.clf()
  plt.figure(figsize=(9, 9))
  sharedX = None
  for i, method in enumerate(METHODS):
    errorsPerSample = data[method]
    allValues = []
    for sample in errorsPerSample.values():
      for err in sample.values():
        if 0 < err['visible']:
          allValues.append(err['dist'])
      continue
    
    ax = plt.subplot(len(METHODS), 1, 1 + i, sharex=sharedX)
    sharedX = ax if sharedX is None else sharedX
    allValues = [min(x, maxDist) for x in allValues]
    plt.hist(allValues, np.arange(maxDist + 1))
    plt.xlabel(TITLES[method])
    continue

  plt.tight_layout()
  return plt

for modelName, methodsInfo in MODELS.items():
  modelDistErrors(methodsInfo).savefig(os.path.join(FOLDER, '%s-dist.png'  % modelName))
  continue

#######################
def scorePredictions(errorsPerSample, clip=1055):
  from sklearn.metrics import mean_squared_error
  ERR = []

  score = 0
  for sample in errorsPerSample.values():
    for err in sample.values():
      if 0 < err['visible']:
        score += min((err['dist'], clip))
        ERR.append(err['dist'])
    continue
  return mean_squared_error(np.zeros_like(ERR), ERR, squared=False)

scoredModels = []
for modelName, methodsInfo in MODELS.items():
  for method, errors in methodsInfo.items():
    scoredModels.append((modelName, method, scorePredictions(errors)))
  continue

scoredModels = sorted(scoredModels, key=lambda x: x[-1])
for x in scoredModels[:11]:
  print(x)
'''  
  plt.clf()
  plt.figure(figsize=(9, 9))
  for i, (metrics, title) in enumerate(metricsWT):
    visP, invisP = metrics['vis'].values()
    plt.subplot(len(metricsWT), 1, 1 + i)
    plt.hist([visP, invisP], 50, label=['Visible', 'Invisible'])
    plt.legend(loc='upper right')
    plt.xlabel(title)
    continue
  
  plt.tight_layout()
  plt.savefig(saveTo('prob.png'))
  return
'''
