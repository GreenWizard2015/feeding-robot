import tensorflow as tf
import tensorflow.keras.backend as K
import NNUtils

class CDetectorTrainer(tf.keras.Model):
  def __init__(self, detector,
    weights,
    masksLoss=NNUtils.diceLoss,
    coordsScale=1.0,
    heatmapsWH=224,
    structuralLoss=True,
    argmaxAccuracy=True,
    adaptiveAccuracyD=5,
    ignoreConfidence=False,
    useAugmentations=False,
    **kwargs
  ):
    super().__init__(**kwargs)
    
    self._detector = detector
    self._masksLoss = masksLoss
    self._coordsScale = float(coordsScale)
    self._heatmapsWH = float(heatmapsWH)
    self._structuralLoss = structuralLoss
    self._argmaxAccuracy = argmaxAccuracy
    self._adaptiveAccuracyD = float(adaptiveAccuracyD)
    self._ignoreConfidence = ignoreConfidence
    self._weights = weights
    self._useAugmentations = useAugmentations
    
    self._loss = tf.keras.metrics.Mean(name="loss")
    self._masksL = tf.keras.metrics.Mean(name="masks")
    self._coordsL = tf.keras.metrics.Mean(name="coords")
    self._probsL = tf.keras.metrics.Mean(name="probs")
    self._consistencyL = tf.keras.metrics.Mean(name="consistency")

    self._adaptiveAccuracy_vis = tf.keras.metrics.Mean(name="accuracy_adaptive_vis")
    self._adaptiveAccuracy_invis = tf.keras.metrics.Mean(name="accuracy_adaptive_invis")
    self._adaptiveAccuracy_th = tf.keras.metrics.Mean(name="accuracy_adaptive_threshold")
    return
  
  def call(self, X, training=False):
    return self._detector([X[0]], training)
  
  @tf.function
  def _visibleMask(self, pos, padding=0.0, keepdims=True):
    mask = tf.math.reduce_all(
      (-padding <= pos) & (pos <= (1.0 + padding)),
      keepdims=keepdims, axis=-1
    )
    return tf.cast(mask, tf.float32)
    
  @tf.function
  def _calcAccuracy(self, y_true, y_pred, thresholdD, thresholdP):
    predPos = y_pred[:, :, :2]
    predProbs = y_pred[:, :, 2:]
    tf.assert_equal(tf.shape(predPos), tf.shape(y_true))

    posScale = self._heatmapsWH
    # calc only if visible
    visibleCoords = self._visibleMask(y_true, padding=0.0, keepdims=False)
    posDistance = NNUtils.vectorLen(y_true - predPos) * visibleCoords * posScale
    tf.assert_equal(tf.shape(posDistance), tf.shape(visibleCoords))
    
    posMatch = posDistance <= thresholdD
    confidentMatch = tf.equal(thresholdP < predProbs[:, :, 0], 0.0 < visibleCoords)
    tf.assert_equal(tf.shape(posMatch), tf.shape(confidentMatch))
    accTotal = tf.cast(tf.logical_and(posMatch, confidentMatch), tf.float32)
    
    accVisible = tf.math.divide_no_nan(
      tf.reduce_sum(accTotal * visibleCoords), tf.reduce_sum(visibleCoords)
    )
    accInvisible = tf.math.divide_no_nan(
      tf.reduce_sum(accTotal * (1.0 - visibleCoords)), tf.reduce_sum(1.0 - visibleCoords)
    )
    return tf.stop_gradient(accVisible), tf.stop_gradient(accInvisible)
  
  @tf.function
  def _accuracy(self, coords, predCoords, predMasks):
    if self._argmaxAccuracy:
      predMasks = NNUtils.channelwise(predMasks)
      maxPt = NNUtils.argmax2d(predMasks)
      P = NNUtils.valueReaderCHW(predMasks)(maxPt)
      predCoords = tf.concat([maxPt, P[..., 0]], axis=-1)
    ####################
    ## adaptive version
    P = predCoords[..., 2]
    visibleCoords = self._visibleMask(coords, padding=0.0, keepdims=False)
    visTh = tf.reduce_min(tf.where(0.0 < visibleCoords, P, 1.0))
    invisTh = tf.reduce_max(tf.where(visibleCoords < 1.0, P, 0.0))
    meanTh = (visTh + invisTh) / 2.0
    
    accVisible, accInvisible = self._calcAccuracy(coords, predCoords, self._adaptiveAccuracyD, meanTh)
    return meanTh, accVisible, accInvisible
  
  @tf.function
  def _posLoss(self, y_true, predPos):
    tf.assert_equal(tf.shape(predPos), tf.shape(y_true))

    dist = NNUtils.vectorLen(y_true - predPos)[..., None] * self._coordsScale
    # calc only if visible
    visibleCoords = self._visibleMask(y_true, padding=0.1, keepdims=False)
    return tf.losses.huber(tf.zeros_like(dist), dist) * visibleCoords
  
  @tf.function
  def _probsLoss(self, coords, predProbs):
    visibleCoords = self._visibleMask(coords, padding=0.1, keepdims=False)[..., None]
    tf.assert_equal(tf.shape(predProbs), tf.shape(visibleCoords))
    return tf.losses.mse(visibleCoords, tf.minimum(predProbs, 1.0))
  
  @tf.function
  def _simpleLoss(self, coords, masks, predCoords, predMasks):
    tf.assert_equal(int(self._heatmapsWH), tf.shape(predMasks)[1])
    tf.assert_equal(int(self._heatmapsWH), tf.shape(predMasks)[2])
    
    masksLoss = self._masksLoss(masks, predMasks)
    coordsLoss = self._posLoss(coords, predCoords[..., :2])
    probsLoss = self._probsLoss(coords, predCoords[..., 2:])
      
    meanTh, accVisible, accInvisible = self._accuracy(coords, predCoords, predMasks)
    return masksLoss, coordsLoss, probsLoss, accVisible, accInvisible, meanTh
    
  @tf.function
  def _consistencyLoss(self, A, B):
    return 1.0 + tf.keras.losses.cosine_similarity(A, B, axis=(-3, -2))
  
  @tf.function
  def _augmentations(self, STATS, images, coords, masks, oldMasks, training):
    consistencyLoss = 0.0
    if not self._useAugmentations: return(consistencyLoss, STATS)

    baseMasks = oldMasks
    FLIPS = [(1, -1), (-1, 1), (-1, -1)]
    for dx, dy in FLIPS:
      predCoords, predMasks = self._detector([images[:, ::dx, ::dy, :]], training=training)
      predMasks = predMasks[:, ::dx, ::dy, :]
      cmask = tf.constant([0.0 if dx < 0 else 1.0, 0.0 if dy < 0 else 1.0, 1.0], predCoords.dtype)
      predCoords = (predCoords * cmask) + ((1.0 - predCoords) * (1.0 - cmask))
      
      STATS = [a + b for a, b in zip(
        STATS,
        self._simpleLoss(coords, masks, predCoords, predMasks)
      )]
      consistencyLoss = consistencyLoss + self._consistencyLoss(baseMasks, predMasks)
      continue
    ###########
    consistencyLoss = consistencyLoss / len(FLIPS)
    STATS = [a / (1 + len(FLIPS)) for a in STATS]
    
    return(consistencyLoss, STATS)
    
  @tf.function
  def _calcLoss(self, data, training):
    (images, confidence), (coords, masks) = data
    predCoords, predMasks = self._detector([images], training=training)
    STATS = self._simpleLoss(coords, masks, predCoords, predMasks)
    consistencyLoss, STATS = self._augmentations(STATS, images, coords, masks, predMasks, training)
    masksLoss, coordsLoss, probsLoss, accVisible, accInvisible, meanTh = STATS 
    
    tf.assert_equal(tf.shape(masksLoss), tf.shape(confidence))
    tf.assert_equal(tf.shape(coordsLoss), tf.shape(confidence))
    tf.assert_equal(tf.shape(probsLoss), tf.shape(confidence))
    
    totalLoss = (
      (masksLoss * self._weights.get('masks', 0.0)) +
      (coordsLoss * self._weights.get('coords', 0.0)) +
      (probsLoss * self._weights.get('probs', 0.0))
    )
    tf.assert_equal(tf.shape(totalLoss), tf.shape(confidence))
    
    if not self._ignoreConfidence:
      totalLoss = totalLoss * confidence
    # consistency loss ignore confidence
    totalLoss = totalLoss + (consistencyLoss * self._weights.get('consistency', 1.0))
    
    if self._structuralLoss:
      totalLoss = tf.reduce_sum(totalLoss, axis=-1) # sum per sample
    ############
    self._masksL.update_state(masksLoss)
    self._coordsL.update_state(coordsLoss)
    self._probsL.update_state(probsLoss)
    self._consistencyL.update_state(consistencyLoss)
    self._loss.update_state(totalLoss)
    
    self._adaptiveAccuracy_vis.update_state(accVisible)
    self._adaptiveAccuracy_invis.update_state(accInvisible)
    self._adaptiveAccuracy_th.update_state(meanTh)
    return totalLoss
  
  def train_step(self, data):
    with tf.GradientTape() as tape:
      totalLoss = self._calcLoss(data, training=True)
    
    TV = self.trainable_variables
    gradients = tape.gradient(totalLoss, TV)
    self.optimizer.apply_gradients(zip(gradients, TV))
    return self._metricsDict()
  
  def test_step(self, data):
    totalLoss = self._calcLoss(data, training=False)
    return self._metricsDict()

  @property
  def metrics(self):
    return [
      self._loss, 
      self._masksL, self._coordsL, self._probsL, self._consistencyL,
      self._adaptiveAccuracy_vis, self._adaptiveAccuracy_invis, self._adaptiveAccuracy_th
    ]

  def _metricsDict(self):
    return {x.name: x.result() for x in self.metrics}
  
  @tf.function
  def evaluateErrorMap(self, images, masks, factor=16, T=1.0):
    eps = K.epsilon()
    neg_mask = 1.0 - masks

    _, predMasks = self._detector([images], training=False)
    diff = (
      -tf.math.log(tf.clip_by_value(1.0 - predMasks, eps, 1.0 - eps))
      * neg_mask
    )
    
    if self._useAugmentations:
      for dx, dy in [(1, -1), (-1, 1), (-1, -1)]:
        _, predMasks = self._detector([images[:, ::dx, ::dy, :]], training=False)
        predMasks = predMasks[:, ::dx, ::dy, :]
        errors = (
          -tf.math.log(tf.clip_by_value(1.0 - predMasks, eps, 1.0 - eps))
          * neg_mask
        )
        diff = tf.maximum(diff, errors)
        continue
      pass
    
    diff = tf.reduce_sum(diff, axis=-1, keepdims=True)
    ###########
    if 1 < factor:
      #diff = tf.nn.avg_pool2d(diff, factor, strides=1, padding='SAME') # smoothing
      diff = tf.nn.max_pool2d(diff, factor, factor, padding='SAME')
      
    return NNUtils.softmax2d(diff / float(T), False)