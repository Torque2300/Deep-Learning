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
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    if probs.ndim == 1:
        subtr = np.zeros_like(probs)
        subtr[target_index] = 1
#         print(subtr,' ',probs,' ',subtr[target_index])
        dprediction = probs - subtr
    else:
        batch_size = predictions.shape[0]
        str_index_arr = np.arange(target_index.shape[0])
        subtr = np.zeros_like(probs)
        subtr[(str_index_arr, target_index.flatten())] = 1
#         print(subtr,' ',probs,' ', subtr[(str_index_arr, target_index.flatten())])
        dprediction = (probs - subtr) / batch_size
    
    return loss, dprediction

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
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dpredictions = softmax_with_cross_entropy(predictions,target_index)
    dW = np.dot(X.T, dpredictions)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

        
    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            loss = 0
            batch_count = len(batches_indices)
            
            for i in range(batch_count):
                s_loss, grad = linear_softmax(X[batches_indices[i]], self.W, y[batches_indices[i]])
                r_loss, r_grad = l2_regularization(self.W, reg)
                s_loss += r_loss
                grad += r_grad
                self.W -= learning_rate * grad
                loss += s_loss
                loss_history.append(s_loss)

            loss = loss / batch_count
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history



    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        probs = softmax(np.dot(X, self.W))
        y_pred = np.argmax(probs, axis = -1) 
        return y_pred