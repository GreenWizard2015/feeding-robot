import numpy as np

class CpcBase:
  def __init__(self, channels, dim, initValue=0.0):
    self._maps = [np.full(dim, initValue, np.float32) for _ in range(channels)]
    return
  
  def update(self, crop, heatmaps):
    return
  
  def result(self):
    return
  
class CpcAvg(CpcBase):
  def __init__(self, channels, dim):
    super().__init__(channels, dim)
    self._probes = np.zeros(dim, np.uint16)
    return
  
  def update(self, crop, heatmaps):
    x1, y1, x2, y2 = crop
    self._probes[x1:x2, y1:y2] += 1
    
    for rmap, hmap in zip(self._maps, heatmaps):
      rmap[x1:x2, y1:y2] += hmap
      continue
    return
  
  def result(self):
    return [x / self._probes for x in self._maps]

class CpcProduct(CpcBase):
  def __init__(self, channels, dim):
    super().__init__(channels, dim, initValue=1.0)
    self._probes = np.zeros(dim, np.uint16)
    return
  
  def update(self, crop, heatmaps):
    x1, y1, x2, y2 = crop
    self._probes[x1:x2, y1:y2] += 1
    
    for rmap, hmap in zip(self._maps, heatmaps):
      rmap[x1:x2, y1:y2] *= 1 + hmap
      continue
    return
  
  def result(self):
    return [np.log2(x) / self._probes for x in self._maps]

class CpcMin(CpcBase):
  def __init__(self, channels, dim):
    super().__init__(channels, dim, initValue=1.0)
    return
  
  def update(self, crop, heatmaps):
    x1, y1, x2, y2 = crop
    for rmap, hmap in zip(self._maps, heatmaps):
      rmap[x1:x2, y1:y2] = np.minimum(rmap[x1:x2, y1:y2], hmap)
      continue
    return
  
  def result(self):
    return [x for x in self._maps]

class CpcMax(CpcBase):
  def __init__(self, channels, dim):
    super().__init__(channels, dim, initValue=0.0)
    return
  
  def update(self, crop, heatmaps):
    x1, y1, x2, y2 = crop
    for rmap, hmap in zip(self._maps, heatmaps):
      rmap[x1:x2, y1:y2] = np.maximum(rmap[x1:x2, y1:y2], hmap)
      continue
    return
  
  def result(self):
    return [x for x in self._maps]

class CpcUncertainty():
  def __init__(self, channels, dim):
    self._N = np.zeros(dim, np.uint16)
    self._mean = [np.zeros(dim, np.float32) for _ in range(channels)]
    self._M2 = [np.zeros(dim, np.float32) for _ in range(channels)]
    return
  
  def update(self, crop, heatmaps):
    x1, y1, x2, y2 = crop
    self._N[x1:x2, y1:y2] += 1
    N = self._N[x1:x2, y1:y2]
    
    for hmap, mean, M2 in zip(heatmaps, self._mean, self._M2):
      delta1 = hmap - mean[x1:x2, y1:y2]
      mean[x1:x2, y1:y2] += delta1 / N
      delta2 = hmap - mean[x1:x2, y1:y2]
      M2[x1:x2, y1:y2] += delta1 * delta2
      continue
    return
  
  def result(self):
    N = self._N
    return [np.sqrt(x / N) for x in self._M2]

PatchCombinersMapping = {
  'avg': CpcAvg,
  'prod': CpcProduct,
  'min': CpcMin,
  'max': CpcMax,
  'uncertainty': CpcUncertainty,
  'std': CpcUncertainty,
}