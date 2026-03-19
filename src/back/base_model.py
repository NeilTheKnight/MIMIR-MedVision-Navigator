import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-7):
    return 1.0 - dice_coef(y_true, y_pred, smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, threshold=0.5, smooth=1e-7):
    y_true_bin = tf.cast(y_true > threshold, tf.float32)
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin)
    union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)

def load_tumor_model(model_path: str):
    custom_objects = {
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss,
        'iou_metric': iou_metric,
    }
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    return model