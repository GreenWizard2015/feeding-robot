import os
import glob

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
DATASET_FOLDER = os.path.join(PROJECT_FOLDER, 'dataset')

SOURCE_FILES = [
  os.path.abspath(x) for x in glob.iglob(os.path.join(PROJECT_FOLDER, '**', '*.avi'), recursive=True)
]

def selectSourceFile():
  if len(SOURCE_FILES) == 1: return SOURCE_FILES[0]
  
  for i, f in enumerate(SOURCE_FILES):
    print('%d | %s' % (i, f))
  
  selected = int(input('Select file:'))
  return SOURCE_FILES[selected]