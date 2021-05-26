import numpy as np
from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy
from copy import copy

train_X, train_y, test_X, test_y = load_svhn("data", max_train=20000, max_test=1000)

knn_classifier = KNN(k=1)
knn_classifier.fit(train_X, train_y)

binary_train_mask = (train_y == 0) | (train_y == 9)
binary_train_X = train_X[binary_train_mask]
binary_train_y = train_y[binary_train_mask] == 0

binary_test_mask = (test_y == 0) | (test_y == 9)
binary_test_X = test_X[binary_test_mask]
binary_test_y = test_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 32*32*3]
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)

# knn_classifier = KNN(k=1)
# knn_classifier.fit(binary_train_X, binary_train_y)

# # TODO: implement compute_distances_two_loops in knn.py
# dists = knn_classifier.compute_distances_two_loops(binary_test_X)
# assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# # TODO: implement compute_distances_one_loop in knn.py
# dists = knn_classifier.compute_distances_one_loop(binary_test_X)
# assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# # TODO: implement compute_distances_no_loops in knn.py
# dists = knn_classifier.compute_distances_no_loops(binary_test_X)
# assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# prediction = knn_classifier.predict(binary_test_X)
# precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
# print("KNN with k = %s" % knn_classifier.k)
# print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

# # Let's put everything together and run KNN with k=3 and see how we do
# knn_classifier_3 = KNN(k=80)
# knn_classifier_3.fit(binary_train_X, binary_train_y)
# prediction = knn_classifier_3.predict(binary_test_X)

# precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
# print("KNN with k = %s" % knn_classifier_3.k)
# print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

num_folds = 10
train_folds_X = np.array_split(binary_train_X, num_folds)
train_folds_y = np.array_split(binary_train_y, num_folds)

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
# k_choices = range(1,50)
k_to_accuracy = {}
k_to_accuracy_2 = {}

for k in k_choices:
    k_to_accuracy[k] = 0
    for fold_i in range(num_folds):

        val_X = train_folds_X[fold_i]
        val_y = train_folds_y[fold_i]

        train_X = np.concatenate([train_folds_X[j] for j in range(num_folds) if j != fold_i])
        train_y = np.concatenate([train_folds_y[j] for j in range(num_folds) if j != fold_i])

        knn_classifier = KNN(k=k)
        knn_classifier.fit(train_X, train_y)
        prediction = knn_classifier.predict(val_X)
        _, _, _, accuracy = binary_classification_metrics(prediction, val_y)

        k_to_accuracy[k] += accuracy
    k_to_accuracy[k] /= num_folds

for k in sorted(k_to_accuracy):
    print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))
