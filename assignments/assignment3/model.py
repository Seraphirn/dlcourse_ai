import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels,
                 conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """

        # self.reg = reg
        # self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        # self.layer2 = ReLULayer()
        # self.layer3 = FullyConnectedLayer(hidden_layer_size, n_output)
        width, height, channels = input_shape
        n_input = conv2_channels * width * height // (16*16)
        self.layers = [
            ConvolutionalLayer(in_channels=channels,
                               out_channels=conv1_channels,
                               filter_size=3,
                               padding=1),
            ReLULayer(),
            MaxPoolingLayer(pool_size=4, stride=4),
            ConvolutionalLayer(in_channels=conv1_channels,
                               out_channels=conv2_channels,
                               filter_size=3,
                               padding=1),
            ReLULayer(),
            MaxPoolingLayer(pool_size=4, stride=4),
            Flattener(),
            FullyConnectedLayer(n_input=n_input, n_output=n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for param in self.params().values():
            param.grad = 0

        layer_input = X

        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = layer_output

        predictions = layer_output

        loss, dy = softmax_with_cross_entropy(predictions, y)

        grad = dy

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        layer_input = X

        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = layer_output

        predictions = layer_output

        pred = np.argmax(predictions, axis=-1)

        return pred

    def params(self):
        result = {}

        for i, layer in enumerate(self.layers):
            for pname, param in layer.params().items():
                result[f'{i+1}-layer {pname}'] = param

        return result
