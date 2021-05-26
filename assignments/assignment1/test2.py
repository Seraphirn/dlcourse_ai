import numpy as np
from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy
from copy import copy

train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
# Now let's use all 10 classes
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# print(train_X.shape)

# knn_classifier = KNN(k=1)
# knn_classifier.fit(train_X, train_y)

# # TODO: Implement predict_labels_multiclass
# predict = knn_classifier.predict(test_X)

# # TODO: Implement multiclass_accuracy
# accuracy = multiclass_accuracy(predict, test_y)
# print("Accuracy: %4.2f" % accuracy)

# Find the best k using cross-validation based on accuracy
NUM_FOLDS = 10
train_folds_X = np.array_split(train_X, NUM_FOLDS)
train_folds_y = np.array_split(train_y, NUM_FOLDS)

# print(list(i.shape for i in train_folds_X))
# print(list(i.shape for i in train_folds_y))

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
# k_choices = range(1,50)
k_to_accuracy = {}

for k in k_choices:
    k_to_accuracy[k] = 0
    for fold_i in range(NUM_FOLDS):

        val_X = train_folds_X[fold_i]
        val_y = train_folds_y[fold_i]

        tmp_train_X = np.concatenate([train_folds_X[j] for j in range(NUM_FOLDS) if j != fold_i])
        tmp_train_y = np.concatenate([train_folds_y[j] for j in range(NUM_FOLDS) if j != fold_i])
        # print(tmp_train_y.shape)
        # print(tmp_train_X.shape)

        # import pdb; pdb.set_trace()

        knn_classifier = KNN(k=k)
        knn_classifier.fit(tmp_train_X, tmp_train_y)
        prediction = knn_classifier.predict(val_X)
        accuracy = multiclass_accuracy(prediction, val_y)

        k_to_accuracy[k] += accuracy
    k_to_accuracy[k] /= NUM_FOLDS

for k in sorted(k_to_accuracy):
    print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))
