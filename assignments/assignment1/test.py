import numpy as np
from gradient_check import check_gradient
import linear_classifer

probs = linear_classifer.softmax(np.array([-5, 0, 5]))
linear_classifer.cross_entropy_loss(probs, 1)


loss, grad = linear_classifer.softmax_with_cross_entropy(
    np.array([1, 0, 0]), 1)
check_gradient(lambda x:
               linear_classifer.softmax_with_cross_entropy(x, 1),
               np.array([1, 0, 0], float))


# TODO Extend combined function so it can receive
# a 2d array with batch of samples
np.random.seed(42)
# Test batch_size = 1
num_classes = 4
batch_size = 1
predictions = np.random.randint(-1, 3,
                                size=(batch_size, num_classes)
                                ).astype(float)
target_index = np.random.randint(0, num_classes,
                                 size=(batch_size)).astype(int)
check_gradient(lambda x:
               linear_classifer.softmax_with_cross_entropy(x, target_index),
               predictions)

exit()

# Test batch_size = 3
num_classes = 4
batch_size = 3
predictions = np.random.randint(-1, 3, size=(batch_size, num_classes)
                                ).astype(float)
target_index = np.random.randint(0, num_classes,
                                 size=(batch_size)).astype(int)
check_gradient(lambda x:
               linear_classifer.softmax_with_cross_entropy(x, target_index),
               predictions)

# Make sure maximum subtraction for numberic stability is done separately
# for every sample in the batch
probs = linear_classifer.softmax(np.array([[20, 0, 0], [1000, 0, 0]]))
assert np.all(np.isclose(probs[:, 0], 1.0))
