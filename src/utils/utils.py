import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

######################################
### Output Weights
######################################
def output_weights(model: nn.Module):
    """
    Prints the trained weights of a given model.

    Args:
        model (nn.Module): The trained model whose weights are to be printed.

    Returns:
        weights (dict): Dictionary containing the weights of the model.
    """
    weights = {}
    for name, param in model.named_parameters():
      if param.requires_grad:
        #print("#" * 20, "\nTrained Model Weights:")
        #print(f"{name}: {param.data}")
        weights[name] = param.data

    return weights


######################################
### Reshape the Input data
######################################
def reshape_input(
    data: np.array,
    n: int,
    sequence_length: int,
    d_model: int
):
    """
    reshapes the input time-series data into a 3D array suitable for feeding into models
    """
    X = np.zeros((n-sequence_length, sequence_length, d_model))

    # fill the first slice of X (single dimension)
    for i in range(n-sequence_length):
        X[i, :, 0] = data[i : i+sequence_length]

    # duplicate across d_model
    for d in range(1, d_model):
        X[:, :, d] = X[:, :, 0]

    Y = data[sequence_length:].reshape(-1, 1)

    return X, Y

######################################
### Creat dataloader
######################################
def split_and_build_dataloader(
    X: np.array,
    Y: np.array,
    batch_size: int,
    split_ratio: float = 0.8,
    shuffle: bool = True
):
    """
    splits data into training and test sets, then builds DataLoaders.

    Args:
        X:                  Input features.
        Y:                  Target labels.
        batch_size:         Batch size for DataLoader.
        split_ratio:        Ratio for splitting the data into train and test sets. Defaults to 0.8.

    Returns:
        dataloader_train:   DataLoader for training data.
        dataloader_test:    DataLoader for test data.
    """
    # Calculate the index for splitting
    n = len(X)
    split_index = int(split_ratio * n)

    # Split the data
    X_train, Y_train = X[:split_index], Y[:split_index]
    X_test, Y_test = X[split_index:], Y[split_index:]

    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)

    # Create TensorDatasets
    dataset_train = TensorDataset(Tensor(X_train), Tensor(Y_train))
    dataset_test = TensorDataset(Tensor(X_test), Tensor(Y_test))

    # Create DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=shuffle)

    return dataloader_train, dataloader_test