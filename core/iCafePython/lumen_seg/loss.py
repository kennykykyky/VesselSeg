import tensorflow as tf
from keras import backend as K


def generic_masked_loss(mask, loss, weights=1, norm_by_mask=True, reg_weight=0, reg_penalty=K.abs):
    def _loss(y_true, y_pred):
        actual_loss = K.mean(mask * weights * loss(y_true, y_pred), axis=-1)
        norm_mask = (K.mean(mask) + K.epsilon()) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = K.mean((1 - mask) * reg_penalty(y_pred), axis=-1)
            return actual_loss / norm_mask + reg_weight * reg_loss
        else:
            return actual_loss / norm_mask

    return _loss


def masked_loss(mask, penalty, reg_weight, norm_by_mask):
    loss = lambda y_true, y_pred: penalty(y_true - y_pred)
    return generic_masked_loss(mask, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)


# TODO: should we use norm_by_mask=True in the loss or only in a metric?
#       previous 2D behavior was norm_by_mask=False
#       same question for reg_weight? use 1e-4 (as in 3D) or 0 (as in 2D)?


def masked_loss_mae(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask)


def masked_loss_mse(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.square, reg_weight=reg_weight, norm_by_mask=norm_by_mask)


def masked_metric_mae(mask):
    def relevant_mae(y_true, y_pred):
        return masked_loss(mask, K.abs, reg_weight=0, norm_by_mask=True)(y_true, y_pred)

    return relevant_mae


def masked_metric_mse(mask):
    def relevant_mse(y_true, y_pred):
        return masked_loss(mask, K.square, reg_weight=0, norm_by_mask=True)(y_true, y_pred)

    return relevant_mse


def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.binary_crossentropy(y_true, y_pred) - K.binary_crossentropy(y_true, y_true), axis=-1)


def split_dist_true_mask(dist_true_mask):
    #tfx,tfy,tfz = tf.split(dist_true_mask, num_or_size_splits=[1, 1, 1], axis=-1)
    tfm = tf.math.reduce_max(dist_true_mask, axis=-1, keepdims=True)
    mask = tf.math.greater(tfm, tf.constant([0], dtype=tf.float32))
    mask = tf.cast(mask, tf.float32)
    return mask, dist_true_mask


def dist_loss(dist_true_mask, dist_pred):
    dist_mask, dist_true = split_dist_true_mask(dist_true_mask)
    masked_dist_loss = masked_loss_mse
    return masked_dist_loss(dist_mask, reg_weight=True)(dist_true, dist_pred)


def relevant_mae(dist_true_mask, dist_pred):
    dist_mask, dist_true = split_dist_true_mask(dist_true_mask)
    return masked_metric_mae(dist_mask)(dist_true, dist_pred)


def relevant_mse(dist_true_mask, dist_pred):
    dist_mask, dist_true = split_dist_true_mask(dist_true_mask)
    return masked_metric_mse(dist_mask)(dist_true, dist_pred)
