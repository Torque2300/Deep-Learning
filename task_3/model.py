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
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layers = [ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
                       ReLULayer(),
                       MaxPoolingLayer(4, 4),
                       ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
                       ReLULayer(),
                       MaxPoolingLayer(4, 4),
                       Flattener(),
                       FullyConnectedLayer(4*conv2_channels, n_output_classes)
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
            param.grad.fill(0.0)
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        data = X.copy()
        for layer in self.layers:
            data = layer.forward(data)
        
        loss, grad = softmax_with_cross_entropy(data, y)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        data = X.copy()
        for layer in self.layers:
            data = layer.forward(data)
        return np.argmax(data, axis = 1)

    def params(self):
        result = {}
        # TODO: Aggregate all the params from all the layers
        # which have parameters       
        name2layer = {"Conv1": self.layers[0],
                      "Conv2": self.layers[3],
                      "Fully": self.layers[7]}

        for name, layer in name2layer.items():
            for k, v in layer.params().items():
                result['{}_{}'.format(name, k)] = v

        return result
