from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

###########################################################
###########################################################
def plot_predictions(
    y_pred: np.array,
    y: np.array,
    split_ratio: int,
    filename: str,
    mse: float
):
    """
    Plots one-step-ahead predictions along with training and test data.

    Args:
        y:                  All target data
        y_pred:             Array of predicted values for the test data.
        split_ratio:        Ratio for splitting the data into train and test sets.
        filename:           Filename to save the plot.

    Returns:
        None
    """
    n = len(y)
    split_index = int(split_ratio * n)

    plt.figure(figsize=(16, 4))

    # Plot training data
    plt.plot(np.arange(split_index), y[:split_index], label='Training Data', color='blue', alpha=0.5)

    # Plot test data
    plt.plot(np.arange(split_index, n), y[split_index:], label='Test Data', color='green', alpha=0.5)

    # Plot predicted values
    plt.plot(np.arange(split_index, n), y_pred, label='Predicted Values', color='red', alpha=0.7)

    # Set titles and labels
    plt.title(f'MSE: {mse:.4f}')
    #plt.xlabel('Time Period')
    plt.legend()
    plt.grid()

    # Save and display plot
    plt.savefig(filename)
    plt.show()

###########################################################
###########################################################
def plot_var_predictions(
    y_pred: np.array,
    y: np.array,
    VaR: None,
    Var_t: None,
    split_ratio: int,
    filename: str,
    mse: float
):
    """
    Plots one-step-ahead predictions along with training and test data.

    Args:
        y:                  All target data
        y_pred:             Array of predicted values for the test data.
        split_ratio:        Ratio for splitting the data into train and test sets.
        filename:           Filename to save the plot.

    Returns:
        None
    """
    n = len(y)
    split_index = int(split_ratio * n)

    plt.figure(figsize=(16, 4))

    # Plot training data
    plt.plot(np.arange(split_index), y[:split_index], label='Training Data', color='blue', alpha=0.5)

    # Plot test data
    plt.plot(np.arange(split_index, n), y[split_index:], label='Test Data', color='green', alpha=0.5)

    # Plot predicted values
    plt.plot(np.arange(split_index, n), y_pred, label='Predicted Values', color='red', alpha=0.7)

    if VaR:
        # Plot VAR data
        plt.plot(np.arange(len(Var_t)), Var_t, label='VaR_t', color='black', alpha=0.5)
        plt.axhline(VaR, color='red', linestyle='--', label='VaR')

    # Set titles and labels
    plt.title(f'MSE: {mse:.4f}')
    #plt.xlabel('Time Period')
    plt.legend()
    plt.grid()

    # Save and display plot
    plt.savefig(filename)
    #plt.show()

###########################################################
###########################################################
def plot_loss(epoch_losses, filename: str):
    """
    Plots the loss values over epochs.

    Args:
        epoch_losses: A list of loss values, one for each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)

    # Save and display plot
    plt.savefig(filename)
    #plt.show()

###########################################################
###########################################################
def display_attention(sentence: list, translation: list, attention: Tensor, 
                      n_heads: int = 8, n_rows: int = 4, n_cols: int = 2):
    """
    Display the attention matrix for each head of a sequence.

    Args:
        sentence:     German sentence to be translated to English; list
        translation:  English sentence predicted by the model
        attention:    attention scores for the heads
        n_heads:      number of heads
        n_rows:       number of rows
        n_cols:       number of columns
    """
    # ensure the number of rows and columns are equal to the number of heads
    assert n_rows * n_cols == n_heads

    # figure size
    fig = plt.figure(figsize=(12,8))

    # visualize each head
    for i in range(n_heads):
        
        # create a plot
        ax = fig.add_subplot(n_rows, n_cols, i+1)
            
        # select the respective head and make it a numpy array for plotting
        _attention = attention.squeeze(0)[i,:,:].cpu().detach().numpy()

        # plot the matrix
        cax = ax.matshow(_attention, cmap='bone')

        # set the size of the labels
        ax.tick_params(labelsize=8)

        # set the indices for the tick marks
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))

        # if the provided sequences are sentences or indices
        if isinstance(sentence[0], str):
            ax.set_xticklabels([t.lower() for t in sentence], rotation=45)
            ax.set_yticklabels(translation)
        elif isinstance(sentence[0], int):
            ax.set_xticklabels(sentence)
            ax.set_yticklabels(translation)

    plt.show()

###########################################################
###########################################################
def display_mask(sentence: list, mask: Tensor, en_itos):
    """
    Display the target mask for each sequence.

    Args:
        sequence:     sequence to be masked
        mask:         target mask for the heads
    """
    # figure size
    fig = plt.figure(figsize=(8,8))
        
    # create a plot
    ax = fig.add_subplot(mask.shape[0], 1, 1)

    # select the respective head and make it a numpy array for plotting
    mask = mask.squeeze(0).cpu().detach().numpy()

    # plot the matrix
    cax = ax.matshow(mask, cmap='bone')

    # set the size of the labels
    ax.tick_params(labelsize=12)

    # set the indices for the tick marks
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))

    # set labels
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel("$Q$")
    ax.set_xlabel("$K^T$")

    if isinstance(sentence[0], int):
        # convert indices to German/English
        sentence = [en_itos[tok] for tok in sentence]

    ax.set_xticklabels(sentence, rotation=75)
    ax.set_yticklabels(sentence)

    plt.show()