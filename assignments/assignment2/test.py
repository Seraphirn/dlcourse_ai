
import numpy as np
# import matplotlib.pyplot as plt

from dataset import load_svhn, random_split_train_val
from gradient_check import check_layer_gradient, check_layer_param_gradient, \
    check_model_gradient
from layers import FullyConnectedLayer, ReLULayer
from model import TwoLayerNet
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from metrics import multiclass_accuracy

def prepare_for_neural_network(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(float) / 255.0

    # Subtract mean
    mean_image = np.mean(train_flat, axis = 0)
    train_flat -= mean_image
    test_flat -= mean_image

    return train_flat, test_flat

train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
train_X, test_X = prepare_for_neural_network(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)

# model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10,
#                     hidden_layer_size = 3, reg = 0)
# loss = model.compute_loss_and_gradients(train_X[:2], train_y[:2])

model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 100, reg = 0)
dataset = Dataset(train_X, train_y, val_X, val_y)
trainer = Trainer(model, dataset, MomentumSGD(), learning_rate=1e-2,
                  num_epochs=100, batch_size=128)

loss_history, train_history, val_history = trainer.fit()

test_pred = model.predict(test_X)
test_accuracy = multiclass_accuracy(test_pred, test_y)
print('Neural net test set accuracy: %f' % (test_accuracy, ))
exit()


learning_rates = [3e-3, 2e-2, 1e-2, 2e-1]
# learning_rates = [1e-4, 1e-2, 1e-1]
# learning_rates = [3e-3]
reg_strength = [1e-4, 2e-4, 3e-2, 1e-1, 2e1, 3]
# reg_strength = [1e-4, 1e-2, 1e-1]
# reg_strength = [1e1]
learning_rate_decay = [0.999, 0.99, 0.9]
# learning_rate_decay = [1]
hidden_layer_size = [128, 64]
# hidden_layer_size = [128]
# hidden_layer_size = [100]
num_epochs = [100]
# batch_size = [64, 128]
batch_size = [128]
optims = [SGD, MomentumSGD]

best_classifier = None
best_val_accuracy = 0

tmp_array = [(lr, rs, lrd, hls, ne, bs, optimm)
             for lr in learning_rates
             for rs in reg_strength
             for lrd in learning_rate_decay
             for hls in hidden_layer_size
             for ne in num_epochs
             for bs in batch_size
             for optimm in optims]

for lr, rs, lrd, hls, ne, bs, optimm in tmp_array:
    print(f'training lrate={lr}, reg={rs}, lrdecay={lrd}, hiddenls={hls}, '
          f'epochs={ne}, batch_size={bs}, {optimm}')
    model = TwoLayerNet(n_input=train_X.shape[1], n_output=10,
                        hidden_layer_size=hls, reg=rs)
    dataset = Dataset(train_X, train_y, val_X, val_y)
    trainer = Trainer(model, dataset, optimm(), learning_rate=lr,
                      num_epochs=ne, learning_rate_decay=lrd,
                      batch_size=bs)
    # model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 100, reg = 1e-1)
    # dataset = Dataset(train_X, train_y, val_X, val_y)
    # trainer = Trainer(model, dataset, SGD(), learning_rate=3e-1, num_epochs=20, batch_size=5)

    try:
        loss_history, train_history, val_history = trainer.fit()
    except Exception as e:
        print(e)
        continue
    if best_val_accuracy < val_history[-1]:

        best_classifier = None
        best_val_accuracy = val_history[-1]
        best_loss_history = loss_history
        best_train_history = train_history
        best_val_history = val_history
        best_hyperparams = lr, rs, lrd, hls, ne, bs, optimm

# TODO find the best hyperparameters to train the network
# Don't hesitate to add new values to the arrays above, perform experiments, use any tricks you want
# You should expect to get to at least 40% of valudation accuracy
# Save loss/train/history of the best classifier to the variables above

print('best validation accuracy achieved: %f' % best_val_accuracy)
print(f'best hyperparams: {best_hyperparams}')
