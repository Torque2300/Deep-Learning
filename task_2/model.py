import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """
    layers = list()
    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        # TODO Create necessary layers
        self.reg = reg
        self.layers = [FullyConnectedLayer(n_input , hidden_layer_size) , 
                       ReLULayer() ,
                       FullyConnectedLayer(hidden_layer_size , n_output)]
       

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.params()['FW'].grad = 0
        self.params()['FB'].grad = 0
        self.params()['SW'].grad = 0
        self.params()['SB'].grad = 0
        
        data = X
        for layer in self.layers:
            data = layer.forward(data)
        loss, d_out = softmax_with_cross_entropy(data, y)
        
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            
        for key in self.params():
            loss_l2, grad_l2 = l2_regularization(self.params()[key].value, self.reg)
            self.params()[key].grad += grad_l2
            loss += loss_l2
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        for key in self.params():
            self.params()[key].grad = 0
        
        data = X
        for layer in self.layers:
            data = layer.forward(data)
            
        pred = np.argmax(data, axis = 1)
        return pred

    def params(self):
        result = {
            'FW':self.layers[0].params()['W'], 'FB':self.layers[0].params()['B'],
            'SW':self.layers[2].params()['W'], 'SB':self.layers[2].params()['B']
        }
        return result