from typing import Tuple
import h5py
import tensorflow as tf
import numpy as np


def load_data(path: str, num: int):
    f = h5py.File(path, 'r')
    return f['data'][:, :num, :], np.squeeze(f['label'][:]).astype(int)


def shuffle(batch, labels):
    i = np.arange(len(labels))
    np.random.shuffle(i)
    return batch[i, ...], labels[i]


def rotate(cloud):
    angle = tf.random.uniform([])*2*np.pi
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    rotm = tf.convert_to_tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return tf.matmul(cloud, rotm)


def jitter(batch, sigma=0.01, clip=0.05):
    jitter = tf.clip_by_value(
        sigma * tf.random.normal(batch.shape), -1*clip, clip)
    return batch+jitter.numpy()
