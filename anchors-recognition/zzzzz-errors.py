#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np

import json
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as hcluster
import math
import glob
from math import floor

_, ax = plt.subplots(3, 2)
sp = []
for x in ax:
  sp.extend(x)
pltID = 0
for fn in glob.glob('d:/*.json'):
  print(fn)
  with open(fn, 'r') as f:
    data = json.load(f)
  
  errors = np.array([np.array(x['distances'], np.float32) for x in data])
  maxError = errors.max(0)
  errors /= maxError
  
  scores = np.array([np.array(x['scores'], np.float32) for x in data])
  scores /= scores.max(0)
  
  combined = np.concatenate((errors, scores), axis=-1)
   
  classes = np.array([(0.85 < x.mean()) for x in scores], np.float32)
  HCInd = np.where(1 == np.array(classes))
  
  transE = TSNE(2, random_state=42).fit_transform(errors[HCInd])
  
  mean = np.mean(transE, axis=0)
  std = np.std(transE, axis=0)
  transE = (transE - mean) / std
#   files = list(set([x['file'] for x in data]))
#   classes = [files.index(x['file']) for x in data]
#   
   
  clusters = hcluster.fclusterdata(transE, .7 , criterion='distance')
  classes = np.array(clusters)
  
  p = sp[pltID]
  pltID += 1
  scatter = p.scatter(
    transE[:, 0], transE[:, 1], c=classes,
    edgecolor='none', alpha=0.5, cmap=plt.cm.nipy_spectral
  )
#   p.legend(*scatter.legend_elements())
  p.title.set_text(fn)
  continue
  
plt.show()