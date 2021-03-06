from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
      # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)
    # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            a = np.random.choice(a = num_train, size = batch_size, replace = True)
            X_batch = X[a]
            y_batch = y[a]
      # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
      # perform parameter update
            self.W -= learning_rate*grad
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        y_pred = np.argmax(X.dot(self.W),axis=1)
        return y_pred
  
    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

