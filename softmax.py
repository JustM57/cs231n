import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):    
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    for i in range(X.shape[0]):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        loss -= np.log(np.exp(scores[y[i]])/sum(np.exp(scores)))
        for j in range(W.shape[1]):
            dW[:, j] += np.exp(scores[j])*X[i]/sum(np.exp(scores))
        dW[:, y[i]] -= X[i]
                       
    loss /= X.shape[0]
    loss += reg * np.sum(W*W) 
    dW /= X.shape[0]
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  # Initialize the loss and gradient to zero.
    dW = np.zeros_like(W)
    scores = X.dot(W)
    scores -= np.max(scores)
    loss = np.sum(-scores[np.arange(X.shape[0]), y] + np.log(np.sum(np.exp(scores), axis=1))) / X.shape[0] + reg * np.sum(W*W) 
    dW = np.dot(X.T, (np.exp(scores).T / np.sum(np.exp(scores), axis = 1)).T)
    y_mat = np.zeros(shape = (X.shape[0], W.shape[1]))
    y_mat[np.arange(X.shape[0]), y] = 1
    dW -= np.dot(X.T, y_mat)
    dW /= X.shape[0]
    return loss, dW
