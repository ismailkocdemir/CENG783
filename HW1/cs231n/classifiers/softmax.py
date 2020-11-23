import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0] #C
  num_train = X.shape[1] #N
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in range(num_train):
    scores = np.dot( W, X[:, i] )
    scores = scores - np.max(scores) 
    probs = np.exp( scores )
    prob_sum = np.sum(probs) 
    softmax = probs / prob_sum
    for j in range(num_classes):    
      dW[j] += (softmax[j])*X[:,i]
      if j == y[i]:
        dW[j] -= X[:,i]
        loss += np.log(softmax[j])
  
  dW = dW / float(num_train)
  dW += reg * W
  loss = 0.5 * reg * np.sum(W * W) + (-1*loss / float(num_train))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0] #C
  num_train = X.shape[1] #N
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  scores = np.dot(W, X)  # CxN
  scores = scores - np.max( scores, axis=0 ) # prevents numeric instability
  softmax = np.exp( scores ) / np.sum( np.exp(scores),  axis = 0 )
  loss =  0.5 * reg * np.sum(W * W) - np.sum( np.log(softmax[y, np.arange(num_train)]) ) / float(num_train)
  
  softmax[ y, np.arange(num_train) ] -= 1
  dW =  reg * W  +  np.dot(softmax, X.T)/ float( num_train)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
