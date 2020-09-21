from tensorflow.keras import backend as K
import tensorflow as tf

class SoftFBeta:


    def __init__(self, beta = 0.17):
        tf.compat.v1.disable_eager_execution()
        self.beta = beta


    def _precision (self, y_true, y_pred):
        return tf.reduce_sum((1 - y_true) * (1 - y_pred)) / (tf.reduce_sum((1 - y_true)) + K.epsilon())


    def _recall (self, y_true, y_pred):
        return tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + K.epsilon())


    def f_beta_soft(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        precision   = self._precision(y_true, y_pred)
        recall      = self._recall(y_true, y_pred)
        f_score     = (1 + K.square(self.beta)) * ((precision * recall) / ((K.square(self.beta) * precision) + recall + K.epsilon()))
        return 1 - f_score