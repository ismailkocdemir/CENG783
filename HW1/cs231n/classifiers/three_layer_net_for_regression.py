import numpy as np
import matplotlib.pyplot as plt


class ThreeLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size[0])
    self.params['b1'] = np.zeros(hidden_size[0])
    self.params['W2'] = std * np.random.randn(hidden_size[0], hidden_size[1])
    self.params['b2'] = np.zeros(hidden_size[1])
    self.params['W3'] = std * np.random.randn(hidden_size[1], output_size)
    self.params['b3'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout = 1.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO:                                                                     #
    # Perform the forward pass, computing the class scores for the input.       #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    hidden1_out = np.dot( X, W1 ) + b1
    relu1_out = np.maximum(0, hidden1_out) 
    drop = (np.random.rand( *relu1_out.shape ) < dropout)  / dropout
    relu1_out *= drop

    hidden2_out = np.dot(relu1_out, W2) + b2
    relu2_out = np.maximum(0, hidden2_out)
    drop = (np.random.rand( *relu2_out.shape ) < dropout)  / dropout
    relu2_out *= drop

    scores = np.dot( relu2_out, W3) + b3
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO:                                                                     #
    # Finish the forward pass, and compute the loss. This should include        #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the squared error     #
    # regression loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    difference = scores - y
    #print 'scores', scores[:5]
    #print 'y', y[:5]
    loss = 0.5*np.sum(np.square( difference))+ 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    grads['W3'] = reg * W3 + np.dot( relu2_out.T , difference )
    grads['b3'] = np.sum( difference, axis = 0) 

    relu2_deriv = np.dot( difference, W3.T ) 
    relu2_deriv[ relu2_out <= 0 ] = 0 # Derivative of ReLU.

    grads['W2'] = reg * W2 + np.dot( relu1_out.T, relu2_deriv )
    grads['b2'] = np.sum( relu2_deriv, axis = 0 )
    
    relu1_deriv = np.dot(relu2_deriv , W2.T ) 
    relu1_deriv[ relu1_out <= 0 ] = 0 # Derivative of ReLU.

    grads['W1'] = reg * W1 + np.dot( X.T, relu1_deriv )
    grads['b1'] = np.sum( relu1_deriv, axis = 0 )

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, dropout = 1.0, momentum=0.9, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    mom = momentum

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_err_history = []
    val_err_history = []

    velocity = {}
    velocity['W2'] = 0.0
    velocity['b2'] = 0.0
    velocity['W1'] = 0.0
    velocity['b1'] = 0.0
    velocity['W3'] = 0.0
    velocity['b3'] = 0.0


    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      random_indexes = np.random.choice(num_train, batch_size)
      X_batch = X[random_indexes]
      y_batch = y[random_indexes]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout=dropout)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      '''
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['W1'] += -learning_rate * grads['W1']
      self.params['b2'] += -learning_rate * grads['b2']
      self.params['W3'] += -learning_rate * grads['W3']
      self.params['b3'] += -learning_rate * grads['b3']
    
      '''
      velocity['W2_prev'] = velocity['W2']
      velocity['W2'] = mom * velocity['W2'] - learning_rate * grads['W2']
      self.params['W2'] += -mom * velocity['W2_prev'] + ( 1 + mom ) * velocity['W2']
      
      velocity['W1_prev'] = velocity['W1']
      velocity['W1'] = mom * velocity['W1'] - learning_rate * grads['W1']
      self.params['W1'] += -mom * velocity['W1_prev'] + ( 1 + mom ) * velocity['W1']
      
      velocity['b2_prev'] = velocity['b2']
      velocity['b2'] = mom * velocity['b2'] - learning_rate * grads['b2']
      self.params['b2'] += -mom * velocity['b2_prev'] + ( 1 + mom ) * velocity['b2']
      
      velocity['b1'] = velocity['b1']
      velocity['b1'] = mom * velocity['b1'] - learning_rate * grads['b1']
      self.params['b1'] += -mom * velocity['b1'] + ( 1 + mom ) * velocity['b1']

      velocity['W3_prev'] = velocity['W3']
      velocity['W3'] = mom * velocity['W3'] - learning_rate * grads['W3']
      self.params['W3'] += -mom * velocity['W3_prev'] + ( 1 + mom ) * velocity['W3']
      
      velocity['b3_prev'] = velocity['b3']
      velocity['b3'] = mom * velocity['b3'] - learning_rate * grads['b3']
      self.params['b3'] += -mom * velocity['b3_prev'] + ( 1 + mom ) * velocity['b3']
     
      #'''
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val error and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check error
        train_err = np.sum(np.square(self.predict(X_batch) - y_batch), axis=1).mean()
        val_err = np.sum(np.square(self.predict(X_val) - y_val), axis=1).mean()
        train_err_history.append(train_err)
        val_err_history.append(val_err)

        # Decay learning rate
        if len(val_err_history) >= 2:
            if  val_err / val_err_history[-2] < 0.5:
                learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_err_history': train_err_history,
      'val_err_history': val_err_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    return self.loss(X, dropout = 1.0)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

