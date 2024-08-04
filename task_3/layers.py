import numpy as np


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

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
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


    def forward(self, X):
        
        batch_size, height, width, channels = X.shape
        self.X = X
        self.X = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        #stride is one. with stride formula is (h(w) - f + 2 * p)/s + 1
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        output = np.zeros([batch_size, out_height, out_width, self.out_channels])

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops    
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                #slice[0,0,0] = np.sum(X[:filter_size,:filter_size,:] * W0) + b0
                #slice[1,0,0] = np.sum(X[1:filter_size +1,:filter_size,:] * W0) + b0
                slice = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :].reshape(batch_size, -1)                
                output[:, y, x, :] = np.dot(slice, self.W.value.reshape(-1, self.out_channels))
                
        return output + self.B.value
                           
    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_in = np.zeros_like(self.X)
        W = self.W.value.reshape(-1, self.out_channels)
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                left = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :].reshape(batch_size, -1).T                
                self.W.grad += np.dot(left, d_out[:, y, x, :]).reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
                d_in[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.dot(d_out[:, y, x, :], W.T).reshape(batch_size,                             self.filter_size, self.filter_size, self.in_channels)
        
        return d_in[:, self.padding:height - self.padding, self.padding:width - self.padding, :]


    def params(self):
        return { 'W': self.W, 'B': self.B }


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
        self.X = X
        
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
        output = np.zeros([batch_size, out_height, out_width, channels])
        for y in range(out_height):
            for x in range(out_width):
                output[:, y, x, :] += np.amax(X[:, y * self.stride:y * self.stride + self.pool_size, x * self.stride:x * self.stride + self.pool_size, :], axis = (1,2))
        return output

               
    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros_like(self.X)
        self.batch_indices = np.arange(batch_size).repeat(channels).reshape((batch_size, channels))
        self.channel_indices = np.arange(channels).repeat(batch_size).reshape((channels, batch_size)).T

        for y in range(out_height):
            for x in range(out_width):
                slice_X = self.X[:,
                          y * self.stride:y * self.stride + self.pool_size,
                          x * self.stride:x * self.stride + self.pool_size,
                          :].reshape(batch_size, -1, channels)
                max_indices = np.unravel_index(np.argmax(slice_X, axis=1), (self.pool_size, self.pool_size))
                dX[self.batch_indices, max_indices[0] + y * self.stride, max_indices[1] + x * self.stride, self.channel_indices] += d_out[:, y, x, :]
        return dX
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
