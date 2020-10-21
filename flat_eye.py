import tensorflow as tf


class FlatEye(tf.keras.initializers.Initializer):
    """Initializer that generates flat identity tensors

    Examples:
    >>> # Standalone usage:
    >>> initializer = FlatEye()
    >>> values = initializer(shape=(9, ))
    >>> # Usage in a Keras layer:
    >>> initializer = FlatEye()
    >>> layer = tf.keras.layers.Dense(3, bias_initializer=initializer)
    """

    def __call__(self, shape, dtype=None):
        """Returns a flat tensor object initialized to identity

        Args:
            shape: Shape of the tensor
            dtype: Optional dtype of the tensor. If not specified,
                `tf.keras.backend.floatx()` is used, which default
                to `float32` unless you configured it otherwise
                (via `tf.keras.backend.set_floatx(float_dtype)`).
        """
        row = tf.math.sqrt(float(shape[0]))
        return tf.constant(tf.eye(row).flatten(), dtype=dtype)
