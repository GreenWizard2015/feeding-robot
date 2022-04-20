import cv2
import numpy as np
from math import ceil
import matplotlib

def VisualizePredictions(
  model,
  img, coords, masks=None,
  IMAGE_PER_ROW=4,
  HW=224,
  zoomTarget=False
):
  images = [None] * IMAGE_PER_ROW
  images[0] = cv2.resize(img * 255., (int(HW), int(HW)))
  
  correctPositions = coords * float(HW)
  
  predPos, predMasks = model.predict(np.array([img]))
  predMasks, predPos = predMasks[0], predPos[0]
  predPositions = predPos[:, :2] * float(HW)
  predProb = predPos[:, 2]

  errors = []
  for i in range(predMasks.shape[-1]):
    mask = np.zeros((int(HW), int(HW), 3))
    if not(masks is None):
      mask[:, :, 2] = cv2.resize(masks[:, :, i] * 255., (int(HW), int(HW)))
    
    mask[:, :, 1] = cv2.resize(predMasks[:, :, i] * 255., (int(HW), int(HW)))

    correctPos = correctPositions[i]
    predPos = predPositions[i]
    if np.any(0 < correctPos) and not zoomTarget:
      anchor = tuple(int(x) for x in correctPos[::-1])
      cv2.circle(mask, anchor, 8, (255, 0, 255), 2)
      
    if zoomTarget:
      x, y = np.clip(predPos - 16, 0, HW - 32).astype(int)
      fragment = mask[x:x+32, y:y+32]
      mask = cv2.resize(fragment, (int(HW), int(HW)))
    else:
      anchor = tuple(int(x) for x in predPos[::-1])
      cv2.circle(mask, anchor, 8, (255, 0, 0), 2)
      
    cv2.putText(
      mask,
      '%.1f (%.1f)' % (predProb[i] * 100.0, predMasks[:, :, i].max() * 100.0), 
      (20, 20), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 0, 0)
    )
    
    maxX, maxY = np.unravel_index(np.argmax(predMasks[..., i]), predMasks[..., i].shape)
    cv2.circle(mask, (maxY, maxX), 8, (255, 255, 255), 2)

    coords = np.stack([correctPos, predPos])
    if np.all(0 <= coords) and np.all(coords <= 224):
      d = np.sqrt((np.subtract(correctPos, predPos) ** 2).sum())
      if 0 < d:
        cv2.putText(mask, 'error = %.1fpx' % d, (20, 210), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255))
        errors.append(d)
        
    if np.all(0 <= correctPos) and np.all(correctPos <= 224):
      maxPos = (maxX, maxY)
      d = np.sqrt((np.subtract(correctPos, maxPos) ** 2).sum())
      if 0 < d:
        cv2.putText(mask, 'error = %.1fpx' % d, (20, 190), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255))
        errors.append(d)
    images.append(mask)
    continue
  
  rows = ceil(len(images) / IMAGE_PER_ROW)
  output = np.ones(((int(HW) + 4) * rows, (int(HW) + 4) * IMAGE_PER_ROW, 3), np.uint8) * 255
  
  for i, img in enumerate(images):
    if not(img is None):
      row = i // IMAGE_PER_ROW
      col = i % IMAGE_PER_ROW
      X = 2 + (int(HW) + 4) * col
      Y = 2 + (int(HW) + 4) * row
      output[Y:Y+int(HW), X:X+int(HW)] = img

  err = (0, 0)
  if errors:
    err = (np.mean(errors), np.std(errors))

  cv2.putText(
    output,
    'Mean error: %.1fpx. Std: %.1f' % err, 
    (int(HW) + 5, 25), cv2.FONT_HERSHEY_COMPLEX, .75, (0, 0, 255)
  )
  return output