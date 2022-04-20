import matplotlib.pyplot as plt
import numpy as np
import Utils, json
from _collections import defaultdict
import random

def ss2dataset(samples):
  return Utils.samples2dataset([(x[0], x[1], x[2]) for x in samples])

def samples2clusters(samples):
  if len(samples) < 5: return [samples]

  errors = np.array([x[-1] for x in samples]).reshape((-1, 1))
  thresh = .02 * (errors.max() - errors.min())

  import scipy.cluster.hierarchy as hcluster
  from collections import Counter
  clustersID = []
  while True:
    clustersID = hcluster.fclusterdata(errors, thresh, criterion="distance")
    cnt = Counter(clustersID)
    ((_, biggestCluster),) = cnt.most_common(1)
    if (biggestCluster <= 1) or (25 < len(cnt)): break
    thresh *= 0.75
  
  clusters = defaultdict(list)
  for cid, sample in zip(clustersID, samples):
    clusters[cid].append(sample)
    continue
  
  clusters = list(clusters.values())
  clusters = list(sorted(clusters, key=lambda x: x[-1][-1]))
  return clusters

def choiceSamples(samplesClusters, N, usedSamples=None, reversePriority=False):
  usedSamples = set() if usedSamples is None else usedSamples
  totalSamples = sum([len(x) for x in samplesClusters])
  assert N <= totalSamples
  clustersP = np.arange(len(samplesClusters), 0, -1) if reversePriority else np.arange(len(samplesClusters))
  clustersP = 2 * clustersP
  
  clustersI = np.arange(len(samplesClusters))
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

with open('d:/samples-e000007.json', 'r') as f:
  samples = json.load(f)
  
values = []
for v in samples.values():
  for x in v.values():
    for xxx in x.values():
      values.append(xxx['confidence'])
    #values.append(x['B1']['confidence'])
    
values = sorted(values)
print(values)
exit()
sample2cluster = {}
clusters = samples2clusters(samples)
for cid, samples in enumerate(clusters):
  print('Cluster %d (%d)' % (cid, len(samples)))
  e = [x[-1] for x in samples]
  print('    ', np.min(e), np.max(e))
  for nm, f, _, err in samples:
    sample2cluster[(nm, f)] = cid
    continue
  continue

print()
spc = defaultdict(int)
samples, _ = choiceSamples(clusters, len(clusters) * 2)
for nm, f, _, err in samples:
  cid = sample2cluster[(nm, f)]
  spc[cid] += 1
  print(nm, f, err, cid)

print()
for c,n in spc.items():
  print(c,n)