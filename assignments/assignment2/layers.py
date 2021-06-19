import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(np.power(W, 2))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # Your final implementation shouldn't have any loops
    # From every batch substact its maximum
    predictions = (predictions.T - np.max(predictions, axis=-1)).T

    exp_preds = np.exp(predictions)
    return (exp_preds.T / np.sum(exp_preds, axis=-1)).T

import math

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if type(target_index) != np.ndarray:
        target_index = np.array([target_index])
    if len(probs.shape) == 1:
        probs = probs[np.newaxis, :]

    result = np.mean(-np.log(
        probs[np.arange(target_index.shape[0]), target_index]
    ))
    if math.isinf(result):
        raise Exception('loss is inf')
    return result


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions
        - gradient of predictions by loss value
    '''
    # Your final implementation shouldn't have any loops
    softmaxf = softmax(predictions)
    loss = cross_entropy_loss(softmaxf, target_index)

    if type(target_index) != np.ndarray:
        target_index = np.array([target_index])

    do_flatten = False

    if len(softmaxf.shape) == 1:
        do_flatten = True
        softmaxf = softmaxf[np.newaxis, :]

    num_samples = softmaxf.shape[0]

    a = tuple([np.arange(target_index.shape[0]), target_index])
    dprediction = softmaxf.copy()

    np.subtract.at(dprediction, a, 1)
    dprediction /= num_samples
    if do_flatten:
        dprediction = dprediction.flatten()

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    def __str__(self):
        return f'value={np.mean(self.value)}; grad={np.mean(self.grad)}'

    def __repr__(self):
        return f'value={np.mean(self.value)}; grad={np.mean(self.grad)}'


class ReLULayer:
    def __init__(self):
        self._X = None

    def forward(self, X):
        self._X = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return d_out * (self._X > 0).astype(int)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # WHY WHY WHY!?!?!?!?!?!?!
        grad = d_out.dot(self.W.value.T)
        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.ones((1, self.X.shape[0])).dot(d_out)

        return grad

    def params(self):
        return {'W': self.W, 'B': self.B}
