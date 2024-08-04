import numpy as np


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
    if predictions.ndim == 1:
        return np.exp(predictions - np.max(predictions))/sum(np.exp(predictions - np.max(predictions)))
    else:
        exp_i = predictions - np.max(predictions, axis=1)[:, np.newaxis]
        exp_sum = np.sum(np.exp(exp_i), axis=1)[:, np.newaxis]
        return np.exp(exp_i) / exp_sum


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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W

    return loss, grad


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
    if probs.ndim == 1:
        return -1*np.log(probs[target_index])
    else:
        target_index = target_index.flatten()
        str_index_arr = np.arange(target_index.shape[0])
        return -np.sum(np.log(probs[(str_index_arr, target_index)])) / target_index.size


def softmax_with_cross_entropy(preds, target_index):
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
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)

    if probs.ndim == 1:
        subtr = np.zeros_like(probs)
        subtr[target_index] = 1
        dprediction = probs - subtr
    else:
        batch_size = preds.shape[0]
        str_index_arr = np.arange(target_index.shape[0])
        subtr = np.zeros_like(probs)
        subtr[(str_index_arr, target_index.flatten())] = 1
        dprediction = (probs - subtr) / batch_size
    
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        self.fwd_data = X
        result = np.copy(X)
        for t in np.nditer(result, op_flags=['readwrite']):
            if t[...] < 0:
                t[...] = 0
        return result

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
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #np.where(self.fwd_data> 0, 1, 0) - это производная функции RELU
        
        return np.where(self.fwd_data > 0, 1, 0) * d_out 

    
    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X,self.W.value) + self.B.value

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
        self.W.grad += np.dot(self.X.T,d_out)
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)
        return np.dot(d_out,self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
