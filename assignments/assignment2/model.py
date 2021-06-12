import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy,\
    l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        # self.layer2 = ReLULayer()
        # self.layer3 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
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

        # for layer in self.layers:
        #     for pname, param in layer.params().items():
        #         if pname == 'W':
        #             reg_loss, reg_dW = l2_regularization(param.value, self.reg)
        #             loss += reg_loss
        #             param.grad += reg_dW

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

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
