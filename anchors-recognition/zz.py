import matplotlib.pyplot as plt
import numpy as np
import Utils, json

def ss2dataset(samples):
  return Utils.samples2dataset([(x[0], x[1], x[2]) for x in samples])

with open('d:/samples-e000003.json', 'r') as f:
  samples = json.load(f)
  
Utils.save('d:/samples-e000003.json', ss2dataset(samples))