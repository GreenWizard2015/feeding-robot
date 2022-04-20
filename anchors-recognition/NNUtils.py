import tensorflow as tf
import tensorflow.keras.backend as K

def softmax2d(x, isChannelwise=True):
  if not isChannelwise:
    x = channelwise(x)
    
  bdim, cdim, xdim, ydim = srcShape = [tf.shape(x)[i] for i in range(4)]
  res = tf.nn.softmax(tf.reshape(x, (bdim * cdim, xdim * ydim)), axis=-1)
  res = tf.reshape(res, srcShape)
  return res if isChannelwise else K.permute_dimensions(res, (0, 2, 3, 1))

def norm2d(x, axis=(-1, -2)):
  maxX = tf.reduce_max(x, axis=axis, keepdims=True)
  return tf.math.divide_no_nan(x, maxX)
  
def flatten(x):
  return tf.reshape(x, (-1,))

def shape(x, N):
  return [tf.shape(x)[i] for i in range(N)]

def valueReaderCHW(values):
  bdim, vdim, xdim, ydim = shape(values, N=4)
  
  ii, jj = tf.meshgrid(tf.range(bdim), tf.range(vdim), indexing='ij')
  ii = tf.reshape(ii, (-1, 1))
  jj = tf.reshape(jj, (-1, 1))
  
  def f(xy):
    xy = tf.clip_by_value(xy, 0.0, 1.0 - K.epsilon())
    xy = tf.multiply(xy, [xdim, ydim])
    cells = tf.cast(tf.floor(xy), tf.int32)
    
    # Make gather index
    x = tf.reshape(cells[..., 0], (-1,))
    y = tf.reshape(cells[..., 1], (-1,))
    
    N = tf.cast(tf.shape(x)[0] / tf.shape(ii)[0], tf.int32)
    bInd = tf.repeat(ii, N, axis=0)
    vInd = tf.repeat(jj, N, axis=0)
    
    indx = tf.stack([flatten(bInd), flatten(vInd), flatten(x), flatten(y)], axis=-1)
    return tf.reshape(tf.gather_nd(values, indx), (bdim, vdim, N, 1))
  return f

def centerOfMassCHW(values):
  bdim, vdim, xdim, ydim = shape(values, N=4)
  
  ii, jj = tf.meshgrid(tf.range(xdim), tf.range(ydim), indexing='ij')
  coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,))], axis=-1)
  coords = tf.cast(coords, tf.float32)
  coords = tf.math.divide(coords, [xdim, ydim])
  
  # Rearrange data into one vector per class
  values_flat = tf.reshape(values, [-1, vdim, xdim * ydim, 1])
  # Compute total mass for each volume
  total_mass = tf.reduce_sum(values_flat, axis=-2)
  # Compute center of mass
  COM = tf.math.divide_no_nan(tf.reduce_sum(values_flat * coords, axis=-2), total_mass)
  COM = tf.clip_by_value(COM, 0.0, 1.0)
  return tf.where(tf.math.is_nan(COM), 0.0, COM)

def channelwise(conv):
  return K.permute_dimensions(conv, (0, 3, 1, 2))

def repeatLast(x, N):
  return tf.repeat(tf.expand_dims(x, -2), repeats=N, axis=-2)

def kl_divergence(p, q):
  return tf.losses.kl_divergence(p, q)
  unsummed_kl = p * (tf.math.log(p + K.epsilon()) - tf.math.log(q + K.epsilon()))
  return tf.reduce_mean(unsummed_kl, -1)

def JensenShannonDivergence(p, q):
  m = (p + q) / 2.0
  return (kl_divergence(p, m) + kl_divergence(q, m)) / 2.0

def diceLoss(y_true, y_pred):
  axes = (1, 2) # [batch dim] + [1, 2] + [classes] 
  numerator = 2. * K.sum(y_pred * y_true, axes)
  denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
  dice = numerator / (denominator + K.epsilon())
  return 1.0 - dice

def log_cosh_dice_loss(y_true, y_pred):
  x = diceLoss(y_true, y_pred)
  return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

def vectorLen(vec):
  return tf.linalg.norm(vec, axis=-1, ord='euclidean')

def focalLoss2d(alpha=2.0, beta=4.0, axis=[-1, -2], toChannelwise=True, threshold=1.0):
  @tf.function
  def focal_loss(hm_true, hm_pred):
    if toChannelwise:
      hm_true = channelwise(hm_true)
      hm_pred = channelwise(hm_pred)
      
    eps = K.epsilon()
    pos_mask = tf.cast(tf.greater_equal(hm_true, threshold), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, threshold), dtype=tf.float32)
    neg_weights = tf.pow(1.0 - hm_true, beta)

    pos_loss = (
      -tf.math.log(tf.clip_by_value(hm_pred, eps, 1.0 - eps))
      * tf.math.pow(1.0 - hm_pred, alpha)
      * pos_mask
    )
    neg_loss = (
        -tf.math.log(tf.clip_by_value(1.0 - hm_pred, eps, 1.0 - eps))
        * tf.math.pow(hm_pred, alpha)
        * neg_weights
        * neg_mask
    )

    num_pos = tf.reduce_sum(pos_mask, axis=axis)
    nonZero = tf.cast(0 < num_pos, tf.float32)
    pos_loss = tf.reduce_sum(pos_loss, axis=axis)
    neg_loss = tf.reduce_sum(neg_loss, axis=axis)

    lossNonZero = tf.math.divide_no_nan(pos_loss + neg_loss, num_pos)
    return tf.where(0 < num_pos, lossNonZero, neg_loss)
  return focal_loss

def argmax2d(values):
  bdim, vdim, xdim, ydim = shape(values, N=4)
  flat = tf.reshape(values, (bdim, vdim, -1))
  indx = tf.cast(tf.argmax(flat, axis=-1), xdim.dtype)
  x = tf.truncatediv(indx, xdim)[..., None]
  y = tf.truncatemod(indx, ydim)[..., None]
  res = tf.concat([x, y], axis=-1)
  return tf.divide(tf.cast(res, tf.float32), [xdim, ydim])
