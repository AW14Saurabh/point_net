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

    def __call__(self, y_true, y_pred, sample_weight=None):
        cce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cl = cce(y_true, y_pred)
        cl = tf.math.reduce_mean(cl)

        md = tf.linalg.matmul(self.ft, tf.transpose(self.ft, perm=[0,2,1]))
        md -= tf.eye(self.ft.shape[1])
        md = tf.nn.l2_loss(md)

        return cl + md * self.rw
