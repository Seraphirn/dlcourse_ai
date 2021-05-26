import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    accuracy = np.sum(prediction==ground_truth) / len(prediction)
    tp = np.sum(np.logical_and(prediction == True, ground_truth == True))
    fp = np.sum(np.logical_and(prediction == True, ground_truth == False))
    fn = np.sum(np.logical_and(prediction == False, ground_truth == True))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2 / (1/accuracy + 1/recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return np.sum(prediction==ground_truth) / len(prediction)
