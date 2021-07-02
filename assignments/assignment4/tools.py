import torch
# from torch.utils.data import Dataset
# import os
# from skimage import io


def _multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return torch.sum(prediction == ground_truth), len(prediction)


def compute_accuracy(model, loader, device):
    """
    Computes accuracy on provided data using mini-batches
    """
    correct_sum = 0
    overall_sum = 0
    for x, y, _ in loader:
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        pred = torch.argmax(model(x_gpu), 1)
        correct, overall = _multiclass_accuracy(pred, y_gpu)
        correct_sum += correct
        overall_sum += overall

    return correct_sum / overall_sum
