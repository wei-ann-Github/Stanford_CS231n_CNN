import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:, y[i]] -= X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W*reg

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  ohe = np.eye(W.shape[1])[y]
  n = y.shape[0]
  scores = X.dot(W)
  loss_mat = ((scores+1).T - np.sum(ohe*scores, axis=1)).T - ohe
  loss = np.sum(loss_mat[loss_mat>=0]) / n
  loss += reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # change W from the XW term.
  # it will be changed by X if it contributes any loss, 
  # i.e. score - score_yi + 1 > 0
  pos_loss = np.array(loss_mat>0, dtype=np.int) # pos_loss has shape [N, C]
  
  # To account for the number of times W_yi is updated with X_yi
  col_sums = np.sum(pos_loss, axis=1).reshape(n,1) # reshape for broadcasting with X

  # W_yi will change if score - score_yi + 1 > 0
  dW = X.T.dot(pos_loss) - (X*col_sums).T.dot(ohe) # ohe has shape [N, C]
  # The first part of dW is to update if SVM_loss for W is > 0.
  # The second part of dW is to update W_yi whenever SVM_loss for W is > 0.

  dW /= X.shape[0] # average dW
  dW += 2*reg*W # take into account the regulariation.
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
