# CS231 Assignment 1 Readme

This assignment is assignment 1 of the Stanford University's course CS231n: Convolutional Neural Networks for Visual Recognition ([Link](http://vision.stanford.edu/teaching/cs231n/index.html)).

The tasks of this assignment is to:

1. Implement a k-nearest neighbour (kNN) classifier where the distance calculation is performed by looping through both the dimensions of the test data and the training data.
    * This is an inefficient method.
* Implement a _predict\_labels_ method for returning the prediction of the kNN.
    * Uses numpy's argsort function
* Implement a kNN classifier where the distance calculation is performed by looping through only the dimensions of the test data.
    * This is less inefficient.
* Implement a kNN classifier where the distance calculation is performed without looping through any of the dimensions of the test data or the training data.
    * This is the efficient vectorized operation.
    * The speed between these 3 implementations of kNN are compared.
* Implement a k-fold cross validation to determine the best hyperparameter, k, for our kNN classifier. This hyperparameter k is the number of nearest neighbour to use to classify our test example.