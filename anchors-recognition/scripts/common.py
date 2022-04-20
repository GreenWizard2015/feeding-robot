import os
import glob
import json

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
DATASET_FILE = os.path.join(PROJECT_FOLDER, 'dataset.json')

SOURCE_FILES = [
  os.path.abspath(x) for x in glob.iglob(os.path.join(PROJECT_FOLDER, '**', '*.avi'), recursive=True)
]

def selectSourceFile(selected=None):
  if len(SOURCE_FILES) == 1: return SOURCE_FILES[0]
  if not (selected is None): return SOURCE_FILES[selected]
  
  for i, f in enumerate(SOURCE_FILES):
    print('%d | %s' % (i, f))
  
  selected = int(input('Select file:'))
  return SOURCE_FILES[selected]

def loadDataset(src=None):
  src = DATASET_FILE if src is None else src
  if os.path.exists(src):
    with open(src) as f:
      return json.load(f)
  return {}