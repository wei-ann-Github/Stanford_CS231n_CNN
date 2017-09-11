Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.
This assignment consists of 5 exercises.

For the exercise on kNN:

* Implement the calculation of distance for k-nearest neighbour (kNN) between 500 test data and 5000 training data using 2 loops. The 1st loop is through the test data and the 2nd loop is a loop through the training data (nested in the test data loop). See cs231n/classifiers/k_nearest_neighbor.py .
* Implement the _predict\_label_ method for returning the kNN label prediction. See cs231n/classifiers/k_nearest_neighbor.py
* Implement the calculation of distance for kNN between 500 test data and 5000 training data using a single loop through the test data. See cs231n/classifiers/k_nearest_neighbor.py
* Implement the calculation of distance for kNN between 500 test data and 5000 training data using a vectorized operations (no looping through the data). See cs231n/classifiers/k_nearest_neighbor.py
* Implement a K-fold validation for fine tuning the hyper-parameter of kNN. The hyper-parameter of kNN is k, the number of nearest neighbour to take reference from to make predictions. See knn.ipynb .

For the exercise on SVM:
* Edited the _svm\_loss\_naive_ function in cs231n/classifiers/linear\_svm.py to calculate the gradient, dW, of the weights, W, for the svm loss function by (_for_) looping through the training example and number of test classes.
* Implemented a vectorized form of the svm loss function to obtain the gradient, dW, of the weights, W, for the svm loss function.
* Implemented SGD in the function LinearClassifier.train() in cs231n/classifiers/linear\_classifier.py
* Wrote the LinearSVM.predict function in cs231n/classifiers/linear\_classifier.py

**Note** There seem to be a mistake in the docstring of linear_classifier.py (line 50) that states that X_batch should have shape (dim, batch_size). However, this dimension is inconsistent in the implementation of the svm_loss function in cs231n/classifiers/linear\_svm.py when X has the shape (n_data, dim). (dim, batch_size) is also inconsistent with the doc string in line 89 of the _predict_ function and line 115 of the _loss_ function in linear_classifier.py . Hence, I have stuck to X_batch with shape = (batch_size, dim).

For the exercise on softmax:
* Implemented the naive softmax loss function with nested loops in cs231n/classifiers/softmax.py
* Implemented the vectorized softmax loss function in cs231n/classifiers/softmax.py
* Fine tuned the linear model using softmax loss function by experimenting with different regularization strength and different learning rate.