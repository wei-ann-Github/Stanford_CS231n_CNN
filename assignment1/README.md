Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.
This assignment consists of 5 exercises.

For the exercise on kNN:

* Implement the calculation of distance for k-nearest neighbour (kNN) between 500 test data and 5000 training data using 2 loops. The 1st loop is through the test data and the 2nd loop is a loop through the training data (nested in the test data loop). See cs231n/classifiers/k_nearest_neighbor.py .
* Implement the _predict\_label_ method for returning the kNN label prediction. See cs231n/classifiers/k_nearest_neighbor.py
* Implement the calculation of distance for kNN between 500 test data and 5000 training data using a single loop through the test data. See cs231n/classifiers/k_nearest_neighbor.py
* Implement the calculation of distance for kNN between 500 test data and 5000 training data using a vectorized operations (no looping through the data). See cs231n/classifiers/k_nearest_neighbor.py
* Implement a K-fold validation for fine tuning the hyper-parameter of kNN. The hyper-parameter of kNN is k, the number of nearest neighbour to take reference from to make predictions. See knn.ipynb .
