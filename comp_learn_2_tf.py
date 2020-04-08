#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.4' # Time-stamp: <2019-05-15T14:29:02Z>

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

input_size = 2
output_size = 2

parser = argparse.ArgumentParser()
parser.add_argument("--max-epoch", default=300, dest="max_epoch", type=int)
parser.add_argument("--max-iters", default=100, dest="max_iters", type=int)
parser.add_argument("--batch-size", default=10, dest="batch_size", type=int)
parser.add_argument("--hidden-size", default=7, dest="hidden_size", type=int)
parser.add_argument("--learning-rate", default=0.1, dest="learning_rate", type=float)
# parser.add_argument("--negative-learning-rate", default=0.1, dest="negative_learning_rate", type=float)
parser.add_argument("--negative-learning-rate", default=0, dest="negative_learning_rate", type=float)
#parser.add_argument("--affine-init", default="0.5", dest="affine_init", choices=["standard", "0.5", "arange"])
parser.add_argument("--affine-init", default="0.5,standard,standard", dest="affine_init")
parser.add_argument("--use-sigmoid", default=False, dest="use_sigmoid", action="store_true")
#parser.add_argument("--use-random-competitor", default=False, dest="use_random_competitor", action="store_true")

args = parser.parse_args()
max_epoch = args.max_epoch
max_iters = args.max_iters
batch_size = args.batch_size
hidden_size = args.hidden_size
learning_rate = args.learning_rate
negative_learning_rate = - args.negative_learning_rate
#affine_init = args.affine_init
use_sigmoid = args.use_sigmoid
#use_random_competitor = args.use_random_competitor
affine_inits = args.affine_init.split(",")
if len(affine_inits) < 2:
    affine_inits.append(affine_inits[0])
if len(affine_inits) < 3:
    affine_inits.append(affine_inits[0])
for s in affine_inits:
    if s not in ["standard", "0.5", "arange", "arange2"]:
        parser.error("--affine-init takes a comma-separated list of {standard,0.5,arange,arange2}.")


## Aurélien Géron『scikit-learnとTensorFlowによる実践機械学習』(下田
## 倫大 監訳, 長尾高弘 訳)を参考にしている。

X = tf.placeholder(tf.float32, shape=(None, input_size), name="X")
y = tf.placeholder(tf.float32, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None, affine_init="0.5"):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs * n_neurons)
        if affine_init == "0.5":
            init = 0.5 * tf.random_normal((n_inputs, n_neurons))
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        elif affine_init == "arange":
            W = tf.Variable(arange_matrix((n_inputs, n_neurons)), name="kernel")
            b = tf.Variable(arange_matrix((n_neurons,)), name="bias")
        elif affine_init == "arange2":
            W = tf.Variable(-arange_matrix((n_inputs, n_neurons)), name="kernel")
            b = tf.Variable(-arange_matrix((n_neurons,)), name="bias")
        else:
            # init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            init = 0.01 * tf.random_normal((n_inputs, n_neurons))
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")

        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

def arange_matrix_tf (shape):
    n = tf.reduce_prod(shape)
    return tf.cast((tf.reshape(tf.range(n), shape) / n) - 0.5, tf.float32)

def arange_matrix (shape):
    n = np.prod(shape)
    return ((np.arange(n).reshape(shape) / n) - 0.5).astype(np.float32)

def answer_of_input(I):
    x0 = I[:,0]
    x1 = I[:,1]
    y0 = x0 ** 2 + x1 ** 2
    y1 = 2 * x0 * x1
    return tf.stack([y0, y1], axis=1)

activation = tf.nn.sigmoid if use_sigmoid else tf.nn.relu
with tf.name_scope("dnn1"):
    hidden1 = neuron_layer(X, hidden_size, "hidden1",
                           activation=activation, affine_init=affine_inits[0])
    hidden2 = neuron_layer(hidden1, hidden_size, "hidden2",
                           activation=activation, affine_init=affine_inits[0])
    outputs1 = neuron_layer(hidden2, output_size, "outputs",
                            affine_init=affine_inits[0])

with tf.name_scope("dnn2"):
    hidden1 = neuron_layer(X, hidden_size, "hidden1",
                           activation=activation, affine_init=affine_inits[1])
    hidden2 = neuron_layer(hidden1, hidden_size, "hidden2",
                           activation=activation, affine_init=affine_inits[1])
    outputs2 = neuron_layer(hidden2, output_size, "outputs",
                            affine_init=affine_inits[1])

with tf.name_scope("dnn3"):
    hidden1 = neuron_layer(X, hidden_size, "hidden1",
                           activation=activation, affine_init=affine_inits[2])
    hidden2 = neuron_layer(hidden1, hidden_size, "hidden2",
                           activation=activation, affine_init=affine_inits[2])
    outputs3 = neuron_layer(hidden2, output_size, "outputs",
                            affine_init=affine_inits[2])
    
# with tf.name_scope("dnn1"):
#     hidden1 = tf.layers.dense(X, hidden_size, "hidden1",
#                               activation=activation)
#     hidden2 = tf.layers.dense(hidden1, hidden_size, "hidden2",
#                               activation=activation)
#     outputs1 = tf.layers.dense(hidden2, output_size, "outputs1")
# 
# with tf.name_scope("dnn2"):
#     hidden1 = tf.layers.dense(X, hidden_size, "hidden1",
#                               activation=activation)
#     hidden2 = tf.layers.dense(hidden1, hidden_size, "hidden2",
#                               activation=activation)
#     outputs2 = tf.layers.dense(hidden2, output_size, "outputs2")

with tf.name_scope("loss1"):
    loss1 = tf.losses.mean_squared_error(labels=y, predictions=outputs1)
with tf.name_scope("loss2"):
    loss2 = tf.losses.mean_squared_error(labels=y, predictions=outputs2)
with tf.name_scope("loss3"):
    loss3 = tf.losses.mean_squared_error(labels=y, predictions=outputs3)

d1 = tf.reduce_sum(tf.square(outputs1 - y), axis=1)
d2 = tf.reduce_sum(tf.square(outputs2 - y), axis=1)
d3 = tf.reduce_sum(tf.square(outputs3 - y), axis=1)
amin = tf.argmin(tf.stack([d1, d2, d3], axis=1), axis=1)
amax = tf.argmax(tf.stack([d1, d2, d3], axis=1), axis=1)
ps = [outputs1, outputs2, outputs3]
y1 = tf.stop_gradient(tf.map_fn(lambda i: tf.gather(tf.gather(ps, tf.gather(amin, i)), i), tf.range(tf.shape(amin)[0]), dtype=tf.float32))
y2 = tf.stop_gradient(tf.map_fn(lambda i: tf.gather(tf.gather(ps, tf.gather(amax, i)), i), tf.range(tf.shape(amax)[0]), dtype=tf.float32))

with tf.name_scope("ploss1"):
    ploss1 = tf.losses.mean_squared_error(labels=y1, predictions=outputs1)
with tf.name_scope("ploss2"):
    ploss2 = tf.losses.mean_squared_error(labels=y1, predictions=outputs2)
with tf.name_scope("ploss3"):
    ploss3 = tf.losses.mean_squared_error(labels=y1, predictions=outputs3)
with tf.name_scope("nloss1"):
    nloss1 = tf.losses.mean_squared_error(labels=y2, predictions=outputs1)
with tf.name_scope("nloss2"):
    nloss2 = tf.losses.mean_squared_error(labels=y2, predictions=outputs2)
with tf.name_scope("nloss3"):
    nloss3 = tf.losses.mean_squared_error(labels=y2, predictions=outputs3)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
noptimizer = tf.train.GradientDescentOptimizer(learning_rate=negative_learning_rate)
training_op1 = optimizer.minimize(ploss1)
training_op2 = optimizer.minimize(ploss2)
training_op3 = optimizer.minimize(ploss3)
ntraining_op1 = noptimizer.minimize(nloss1)
ntraining_op2 = noptimizer.minimize(nloss2)
ntraining_op3 = noptimizer.minimize(nloss3)

answer = answer_of_input(X)

init = tf.global_variables_initializer()

total_loss1 = 0
total_loss2 = 0
total_loss3 = 0
loss_count = 0
loss_list1 = []
loss_list2 = []
loss_list3 = []

with tf.Session() as sess:
    init.run()

    for epoch in range(max_epoch):
        for iters in range(max_iters):
            X_val = np.random.uniform(-1.0, 1.0, (batch_size, input_size))
            y_val = sess.run(answer, feed_dict={X: X_val})
            loss1_val, loss2_val, loss3_val, _, _, _, _, _, _ \
                = sess.run((loss1, loss2, loss3,
                            training_op1, training_op2, training_op3,
                            ntraining_op1, ntraining_op2, ntraining_op3),
                           feed_dict={X: X_val, y: y_val})

            total_loss1 += loss1_val
            total_loss2 += loss2_val
            total_loss3 += loss3_val
            loss_count += 1

            if (iters + 1) % 10 == 0:
                avg_loss1 = total_loss1 / loss_count
                avg_loss2 = total_loss2 / loss_count
                avg_loss3 = total_loss3 / loss_count
                print('| epoch %d | iter %d / %d | loss %.2f, %.2f, %.2f'
                      % (epoch + 1, iters + 1, max_iters, avg_loss1, avg_loss2, avg_loss3))
                loss_list1.append(avg_loss1)
                loss_list2.append(avg_loss2)
                loss_list3.append(avg_loss3)
                total_loss1, total_loss2, total_loss3, loss_count = 0, 0, 0, 0

ylim = None
x = np.arange(len(loss_list1))
if ylim is not None:
    plt.ylim(*ylim)
plt.plot(x, loss_list1, label='train1')
plt.plot(x, loss_list2, label='train2')
plt.plot(x, loss_list3, label='train3')
plt.xlabel('iterations (x' + str(10) + ')')
plt.ylabel('loss')
plt.legend()
plt.show()
