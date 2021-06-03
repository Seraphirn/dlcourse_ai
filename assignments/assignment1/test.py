import numpy as np

from dataset import load_svhn, random_split_train_val
# from gradient_check import check_gradient
from metrics import multiclass_accuracy
import linear_classifer


def prepare_for_linear_classifier(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(float) / 255.0

    # Subtract mean
    mean_image = np.mean(train_flat, axis=0)
    train_flat -= mean_image
    test_flat -= mean_image

    # Add another channel with ones as a bias term
    train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0],
                                                           1))])
    test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0],
                                                         1))])
    return train_flat_with_ones, test_flat_with_ones


num_epochs = 200
batch_size = 300

learning_rates = [1e-3, 1e-4, 1e-5]
reg_strengths = [1e-4, 1e-5, 1e-6]

train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000,
                                             max_test=1000)
train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y,
                                                        num_val=1000)

best_classifier = None
best_val_accuracy = None
for lrate, reg_str in zip(learning_rates, reg_strengths):

    classifier = linear_classifer.LinearSoftmaxClassifier()
    loss_history = classifier.fit(train_X, train_y, epochs=num_epochs,
                                  learning_rate=lrate, batch_size=batch_size,
                                  reg=reg_str)
    pred = classifier.predict(val_X)
    accuracy = multiclass_accuracy(pred, val_y)
    if best_val_accuracy is None or best_val_accuracy < accuracy:
        best_classifier = classifier
        best_val_accuracy = accuracy

print('best validation accuracy achieved: %f' % best_val_accuracy)
