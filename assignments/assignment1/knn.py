import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import cdist

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = np.sum(
                    np.abs(X[i_test] - self.train_X[i_train])
                )
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)

        for i_test in range(num_test):
            dists[i_test] = np.sum(
                np.abs(self.train_X - X[i_test]),
                axis=1
            )
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''

        # this works slowly - like 2 loop in python
        # return np.sum(
        #     np.abs(X[:,np.newaxis] - self.train_X[np.newaxis,:]),
        #     axis=2,
        #     dtype=np.float32
        # )
        return cdist(X, self.train_X, metric='cityblock')

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        '''
        args_of_k_near = np.argpartition(dists, kth=self.k, axis=1)[:,:self.k]
        labels_of_k_near = self.train_y[args_of_k_near]
        predicts = np.array(mode(labels_of_k_near, axis=1),
                            np.bool)[0,:].flatten()
        return predicts

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        '''

        args_of_k_near = np.argpartition(dists, kth=self.k, axis=1)[:,:self.k]
        labels_of_k_near = self.train_y[args_of_k_near]
        predicts = np.array(mode(labels_of_k_near, axis=1))[0,:].flatten()

        return predicts
