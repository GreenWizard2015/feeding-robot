#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np

import json
import matplotlib.pyplot as plt
import Utils
'''
with open('metric_history.json', 'r') as f:
  metrics = json.load(f)

loss, valLoss = metrics['loss'], metrics['val_loss']
accuracy, valAccuracy = metrics['accuracy'], metrics['val_accuracy']

def adjustPlot(values, shift, axes, d):
  axes.set_ylim(
    min([np.min(x[shift:]) for x in values]) - d,
    max([np.max(x[shift:]) for x in values]) + d
  )
  return

plt.plot(np.arange(len(loss)), loss, label='train')
plt.plot(np.arange(len(valLoss)), valLoss, label='validation')
adjustPlot([loss, valLoss], shift=50, axes=plt.gca(), d=1)
plt.legend()
plt.show()

loss, valLoss = metrics['accuracy'], metrics['val_accuracy']
plt.plot(np.arange(len(loss)), loss, label='train')
plt.plot(np.arange(len(valLoss)), valLoss, label='validation')
adjustPlot([loss, valLoss], shift=50, axes=plt.gca(), d=.05)
plt.legend()
plt.show()
'''
# '''
with open('d:/mined-samples-x.json', 'r') as f:
  d = json.load(f)
  
samples = Utils.flattenDataset(d)
samples = Utils.normalizeConfidence(samples)

filtered = []
for A, B, points in samples:
  save = any((0.9 < x.get('confidence', 1)) for x in points.values())
  if save:
    filtered.append((A, B, points))
samples = filtered
Utils.save('d:/x.json', Utils.samples2dataset(samples))
confidences = [Utils.confidence(x[-1]) for x in samples]
print(confidences)
plt.hist(confidences, bins=50)
# plt.gca().set_ylim(0, 11)
plt.show()
exit()
# '''

values = [
  [.3, 0, 0],
  [.36, .338, .6],
  [.61, .62, .6],
  [.61, .62, .6],
]

for i in range(3):
  xval = [j+i*.2 for j, v in enumerate(values)]
  yval = [v[i] for v in values]
  plt.bar(xval, yval, .1)
  
plt.gca().set_ylabel('Accuracy')
plt.gca().set_xlabel('Models')
plt.xticks(.2+np.arange(len(values)), ['%d iteration' % i for i,m in enumerate(values)])
plt.show()