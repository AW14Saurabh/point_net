import tensorflow as tf
from tensorflow.keras.losses import Loss


class PointLoss(Loss):
    """Computes sparse softmax cross entropy between `logits` and `labels`.

        To it the l2 loss of the orthogonal matrix of the feature transform 
        is added.
    """
    def __init__(self, ft) -> None:
        self.ft = ft
        self.rw = 0.001

    def call(self, y_true, y_pred):
        cl = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cl = tf.math.reduce_mean(cl)

        md = tf.linalg.matmul(self.ft, tf.transpose(self.ft, perm=[0,2,1]))
        md -= tf.constant(tf.eye(self.ft.shape[1]), dtype=tf.float32)
        md = tf.nn.l2_loss(md)

        return cl + md * self.rw
