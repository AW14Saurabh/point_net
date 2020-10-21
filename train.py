import tensorflow as tf
from tensorflow.keras import layers
from flat_eye import FlatEye
from loss import PointLoss
from data import jitter, load_data, rotate, shuffle
# Data
x_train, y_train = load_data('data/modelnet40_ply_hdf5_2048/ply_data_train0.h5')
x_test, y_test = load_data('data/modelnet40_ply_hdf5_2048/ply_data_test0.h5')
x_train = x_train[:,:1024,:]
x_test = x_test[:,:1024,:]
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

# Input
inputs = tf.keras.Input(shape=(1024,3),batch_size=32, name='pointclouds', dtype=tf.float32)
inputs = tf.map_fn(rotate, inputs)
inputs = jitter(inputs)

# Input Transform
inp = tf.expand_dims(inputs, -1)
it = layers.Conv2D(64, (1, 3), activation='relu', kernel_regularizer='l2')(inp)
it = layers.BatchNormalization()(it)
it = layers.Conv2D(128, (1, 1), activation='relu', kernel_regularizer='l2')(it)
it = layers.BatchNormalization()(it)
it = layers.Conv2D(1024, (1, 1), activation='relu', kernel_regularizer='l2')(it)
it = layers.BatchNormalization()(it)
it = layers.MaxPool2D((inputs.shape[1], 1), (2, 2))(it)
it = tf.reshape(it, (inputs.shape[0], -1))
it = layers.Dense(512, activation='relu', kernel_regularizer='l2')(it)
it = layers.BatchNormalization()(it)
it = layers.Dense(256, activation='relu', kernel_regularizer='l2')(it)
it = layers.BatchNormalization()(it)
it = layers.Dense(9, kernel_initializer='zeros', bias_initializer=FlatEye())(it)
it = tf.reshape(it, (inputs.shape[0], 3, 3))

# Model mlp(64, 64)
inp = tf.linalg.matmul(inputs, it)
inp = tf.expand_dims(inp, -1)
net = layers.Conv2D(64, (1, 3), activation='relu', kernel_regularizer='l2')(inp)
net = layers.BatchNormalization()(net)
net = layers.Conv2D(64, (1, 1), activation='relu', kernel_regularizer='l2')(net)
net = layers.BatchNormalization()(net)

# Feature Transform
ft = layers.Conv2D(64, (1, 1), activation='relu', kernel_regularizer='l2')(net)
ft = layers.BatchNormalization()(ft)
ft = layers.Conv2D(128, (1, 1), activation='relu', kernel_regularizer='l2')(ft)
ft = layers.BatchNormalization()(ft)
ft = layers.Conv2D(1024, (1, 1), activation='relu', kernel_regularizer='l2')(ft)
ft = layers.BatchNormalization()(ft)
ft = layers.MaxPool2D((net.shape[1], 1), (2, 2))(ft)
ft = tf.reshape(ft, (net.shape[0], -1))
ft = layers.Dense(512, activation='relu', kernel_regularizer='l2')(ft)
ft = layers.BatchNormalization()(ft)
ft = layers.Dense(256, activation='relu', kernel_regularizer='l2')(ft)
ft = layers.BatchNormalization()(ft)
ft = layers.Dense(4096, kernel_initializer='zeros', bias_initializer=FlatEye())(ft)
ft = tf.reshape(ft, (net.shape[0], 64, 64))

# Model mlp(64, 128, 1024)
net = tf.linalg.matmul(tf.squeeze(net, axis=[2]), ft)
net = tf.expand_dims(net, [2])
net = layers.Conv2D(64, (1, 1), activation='relu', kernel_regularizer='l2')(net)
net = layers.BatchNormalization()(net)
net = layers.Conv2D(128, (1, 1), activation='relu', kernel_regularizer='l2')(net)
net = layers.BatchNormalization()(net)
net = layers.Conv2D(1024, (1, 1), activation='relu', kernel_regularizer='l2')(net)
net = layers.BatchNormalization()(net)

# Max Pool
net = layers.MaxPool2D((inputs.shape[1], 1), (2, 2))(net)
net = tf.reshape(net, (inputs.shape[0], -1))

# Model mlp(512, 256, 40)
net = layers.Dense(512, activation='relu', kernel_regularizer='l2')(net)
net = layers.BatchNormalization()(net)
net = layers.Dropout(0.3)(net)
net = layers.Dense(256, activation='relu', kernel_regularizer='l2')(net)
net = layers.BatchNormalization()(net)
net = layers.Dropout(0.3)(net)
outputs = layers.Dense(40, kernel_regularizer='l2')(net)

# Build
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 200000, 0.7, True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PointNet')
model.compile(optimizer=optimizer, loss=PointLoss(ft), metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=250)
model.evaluate(x_test, y_test)