import h5py
import tensorflow as tf
import math

def load_data(path):
    f = h5py.File(path, 'r')
    return tf.convert_to_tensor(f['data'][:]), tf.squeeze(f['label'][:])

def shuffle(x, y):
    i = tf.random.shuffle(tf.range(len(y)))
    return x[i, ...], y[i]

def rotate(cloud):
    angle = tf.random.uniform([])*2*math.pi
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    rotm = tf.convert_to_tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return tf.matmul(cloud, rotm)

def jitter(batch, sigma=0.01, clip=0.05):
    jitter = tf.clip_by_value(sigma * tf.random.normal(batch.shape), -1*clip, clip)
    return batch+jitter