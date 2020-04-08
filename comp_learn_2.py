#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.4' # Time-stamp: <2019-05-15T14:28:36Z>

import numpy as np
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

## 斎藤康毅『ゼロから作る Deep Leaning』シリーズのソースをコピペしたり、
## 参考にしたりしている。レポジトリは↓。
## https://github.com/oreilly-japan/deep-learning-from-scratch
## https://github.com/oreilly-japan/deep-learning-from-scratch-2
## -- 引用: はじめ --

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

## -- 引用: おわり --

class ReLU:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class IdentityWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = x

        loss = np.sum((x - t) ** 2) / np.prod(t.shape)
        return loss

    def backward(self, dout=1):
        dx = 2 * (self.y - self.t) * dout / np.prod(self.t.shape)
        return dx

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, affine_init="0.5"):
        I, H, O = input_size, hidden_size, output_size

        if affine_init == "0.5":
            W1 = 0.5 * np.random.randn(I, H)
            b1 = np.zeros(H)
            W2 = 0.5 * np.random.randn(H, H)
            b2 = np.zeros(H)
            W3 = 0.5 * np.random.randn(H, O)
            b3 = np.zeros(O)
        elif affine_init == "arange":
            W1 = arange_matrix((I, H))
            b1 = arange_matrix((H,))
            W2 = arange_matrix((H, H))
            b2 = arange_matrix((H,))
            W3 = arange_matrix((H, O))
            b3 = arange_matrix((O,))
        elif affine_init == "arange2":
            W1 = -arange_matrix((I, H))
            b1 = -arange_matrix((H,))
            W2 = -arange_matrix((H, H))
            b2 = -arange_matrix((H,))
            W3 = -arange_matrix((H, O))
            b3 = -arange_matrix((O,))
        else:
            W1 = 0.01 * np.random.randn(I, H)
            b1 = np.zeros(H)
            W2 = 0.01 * np.random.randn(H, H)
            b2 = np.zeros(H)
            W3 = 0.01 * np.random.randn(H, O)
            b3 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid() if use_sigmoid else ReLU(),
            Affine(W2, b2),
            Sigmoid() if use_sigmoid else ReLU(),
            Affine(W3, b3)
        ]
        self.loss_layer = IdentityWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calc_loss(self, score, t):
        loss = self.loss_layer.forward(score, t)
        return loss

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.calc_loss(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
                
def arange_matrix (shape):
    n = np.prod(shape)
    return (np.arange(n).reshape(shape) / n) - 0.5

def answer_of_input(I):
    x0 = I[:,0]
    x1 = I[:,1]
    y0 = x0 ** 2 + x1 ** 2
    y1 = 2 * x0 * x1
    return np.c_[y0, y1]

model1 = ThreeLayerNet(input_size=input_size, hidden_size=hidden_size,
                       output_size=output_size, affine_init=affine_inits[0])
model2 = ThreeLayerNet(input_size=input_size, hidden_size=hidden_size,
                       output_size=output_size, affine_init=affine_inits[1])
model3 = ThreeLayerNet(input_size=input_size, hidden_size=hidden_size,
                       output_size=output_size, affine_init=affine_inits[2])
optimizer = SGD(lr=learning_rate)
noptimizer = SGD(lr=negative_learning_rate)

total_loss1 = 0
total_loss2 = 0
total_loss3 = 0
loss_count = 0
loss_list1 = []
loss_list2 = []
loss_list3 = []

for epoch in range(max_epoch):
    for iters in range(max_iters):
        batch_x = np.random.uniform(-1.0, 1.0, (batch_size, input_size))
        batch_t = answer_of_input(batch_x)

        p1 = model1.predict(batch_x)
        p2 = model2.predict(batch_x)
        p3 = model3.predict(batch_x)
        d1 = np.sum(np.square(p1 - batch_t), axis=1)
        d2 = np.sum(np.square(p2 - batch_t), axis=1)
        d3 = np.sum(np.square(p3 - batch_t), axis=1)
        amin = np.argmin(np.c_[d1, d2, d3], axis=1)
        amax = np.argmax(np.c_[d1, d2, d3], axis=1)
        ps = [p1, p2, p3]
        t1 = np.array([ps[p][i] for i, p in enumerate(amin)])
        t2 = np.array([ps[p][i] for i, p in enumerate(amax)])
        ploss1 = model1.calc_loss(p1, t1)
        Ploss2 = model2.calc_loss(p2, t1)
        Ploss3 = model3.calc_loss(p3, t1)
        model1.backward()
        model2.backward()
        model3.backward()
        optimizer.update(model1.params, model1.grads)
        optimizer.update(model2.params, model2.grads)
        optimizer.update(model3.params, model3.grads)

        nloss1 = model1.calc_loss(p1, t2)
        nloss2 = model2.calc_loss(p2, t2)
        nloss3 = model2.calc_loss(p3, t2)
        model1.backward()
        model2.backward()
        model3.backward()
        noptimizer.update(model1.params, model1.grads)
        noptimizer.update(model2.params, model2.grads)
        noptimizer.update(model3.params, model3.grads)
        
        loss1 = model1.calc_loss(p1, batch_t)
        loss2 = model2.calc_loss(p2, batch_t)
        loss3 = model3.calc_loss(p3, batch_t)

        total_loss1 += loss1
        total_loss2 += loss2
        total_loss3 += loss3
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
