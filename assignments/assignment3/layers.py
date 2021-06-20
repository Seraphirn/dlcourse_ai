import numpy as np
import math


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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
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


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = 1

    def forward(self, X):
        batch_size, width, height,  channels = X.shape
        self.X = X

        # left filter border
        left_fb = (self.filter_size-1)//2
        right_fb = self.filter_size//2

        out_height = (height - self.filter_size
                      + 2*self.padding) // self.stride + 1
        out_width = (width - self.filter_size
                     + 2*self.padding) // self.stride + 1

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        Y = np.zeros(shape=(batch_size, out_width, out_height,
                            self.out_channels))

        for x in range(out_width):
            for y in range(out_height):
                receptive_field = X[:, x-left_fb:x+right_fb+1,
                                    y-left_fb:y+right_fb+1, :]
                Y[:, x, y, :] = receptive_field.reshape((batch_size, -1)).dot(
                    self.W.value.reshape((-1, self.out_channels))
                ) + self.B.value

        return Y

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # left filter border
        left_fb = (self.filter_size-1)//2
        right_fb = self.filter_size//2

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        grad = np.zeros(shape=self.X.shape)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                # WHY WHY WHY!?!?!?!?!?!?!
                tmp = d_out[:, x, y, :].dot(
                    self.W.value.reshape((-1, self.out_channels)).T
                )
                grad[:, x-left_fb:x+right_fb+1, y-left_fb:y+right_fb+1, :] +=\
                    tmp.reshape((batch_size, self.filter_size,
                                 self.filter_size, channels))

                recept_field = self.X[:, x-left_fb:x+right_fb+1,
                                      y-left_fb:y+right_fb+1, :]
                recept_field = recept_field.reshape((batch_size, -1))

                self.W.grad += recept_field.T.dot(d_out[:, x, y, :]).reshape(
                    self.W.value.shape
                )
                self.B.grad += np.ones((1, batch_size))\
                    .dot(d_out[:, x, y, :])\
                    .reshape((self.out_channels))
        return grad

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
