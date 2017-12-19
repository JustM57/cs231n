from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        ReLU = lambda x: np.maximum(0, x)
        softmax = lambda x: 1/(1 + np.exp(-x))
        hidden = ReLU(X.dot(W1) + b1)
        scores = hidden.dot(W2) + b2
        H = hidden.shape[0]
  
    # If the targets are not given then jump out, we're done
        if y is None:
            return scores

    # Compute the loss
        scores -= np.max(scores)
        loss = np.sum(-scores[np.arange(N), y] + np.log(np.sum(np.exp(scores), axis=1))) / N 
        loss += reg * (np.sum(W1*W1) + np.sum(W2*W2)) 
    # Backward pass: compute gradients
        grads = {}
        dW2 = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims=True)
        y_mat = np.zeros(shape = (N, W2.shape[1]))
        y_mat[np.arange(N), y] = 1
        dW2 = (dW2 - y_mat) / N
        db2 = np.sum(dW2, axis = 0, keepdims = True)
        dhidden = np.dot(dW2, W2.T)
        dW2 = np.dot(hidden.T, dW2)
        dW2 += 2*reg*W2
        grads['W2'] = dW2
        grads['b2'] = db2
        dW1 = dhidden
        dW1[hidden == 0] = 0
        db1 = np.sum(dW1, axis=0, keepdims=True)
        dW1 = np.dot(X.T, dW1)
        dW1 += 2*reg*W1
        grads['W1'] = dW1
        grads['b1'] = db1
        
        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            a = np.random.choice(a=num_train, size=batch_size, replace=True)
            X_batch = X[a]
            y_batch = y[a]
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1'].reshape(grads['b1'].shape[1], )
            self.params['b2'] -= learning_rate * grads['b2'].reshape(grads['b2'].shape[1], )
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        ReLU = lambda x: np.maximum(0, x)
        hidden = ReLU(X.dot(self.params['W1']) + self.params['b1'])
        scores = hidden.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        return y_pred


