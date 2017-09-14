import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]; num_class = W.shape[1];
  for i in range(num_train):
        score_i = np.exp(X[i].dot(W))
        score_correct_class = score_i[y[i]];
        total_score = np.sum(score_i)
        loss -= np.log(score_correct_class/total_score)
        for j in range(num_class):
            dW[:,j] += score_i[j]/total_score*X[i]
            if j == y[i]:
                dW[:,j] -= X[i]
  dW /= num_train
  loss /= num_train
    
  # regularization
  dW += 2*reg*W
  loss += reg*np.sum(W**2)
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  indicator = np.eye(W.shape[1])[y] # shape = (N, C)
  scores = np.exp(X.dot(W)) # shape = (N, C)
  total_score = np.sum(scores, axis=1) # shape = like (1, N)
  correct_class_score = scores * indicator # shape = (N, C)
  
  # loss + regularization (don't forget to log p)
  loss = np.mean(-np.log(np.sum(correct_class_score, axis=1)/total_score))
  loss += reg * np.sum(W**2) 

  # dW + derivative of regularization
  dW = X.T.dot(scores/total_score.reshape(scores.shape[0], 1)) - X.T.dot(indicator)
  dW /= X.shape[0]
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW