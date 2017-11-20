import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):

    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    num_train = X.shape[0]
    scores = X.dot(W)
    margins = scores + 1 - scores[np.arange(num_train), y].reshape(num_train, 1)
    margins[margins < 0] = 0
    margins[np.arange(num_train), y] = 0 
    loss = np.sum(margins)/num_train
    loss += reg * np.sum(W * W)
    margins[margins > 0] = 1
    margins[np.arange(num_train), y] -= np.sum(margins, axis=1).T
    dW = np.dot(X.T, margins)
    dW /= num_train

    return loss, dW
