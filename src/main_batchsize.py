
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.attention import MultiHeadAttention, MultiHeadAttention2
from models.feed_forward import PositionwiseFeedForward
from models.encoder import Encoder
from models.decoder import Decoder

from train_and_test.train_and_test import train_model, evaluate_model
from utils.utils import output_weights, reshape_input, split_and_build_dataloader
from utils.visualize import plot_predictions, plot_loss

#seed = 2024
#seed = 123
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

###########################################################
### Data Generation Process
###########################################################
phi = 0.4
sigma = 0.2  # sd. of noise
n = 1000  # number of time points

y = np.zeros(n)
epsilon = np.random.normal(0, sigma, n)
y[0] = epsilon[0] # initialize the first value of the series

# generate AR(1) time series
for t in range(1, n):
    y[t] = phi * y[t-1] + epsilon[t]
    #y_ar1[t] = np.sin(y_ar1[t-1]) + epsilon[t]

###########################################################
### Model
###########################################################
class TimeSeriesAttention(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        d_model: int = 512,
        d_ffn: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias_att: bool = False
    ):
        super().__init__()
        #self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias_att)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, bias=bias_att, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.linear1 = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(sequence_length, 1)
        self.relu = nn.ReLU()

    def forward(
        self,
        x: Tensor,
        mask_or_not: bool = False
    ):
        if mask_or_not == True:
          mask_matrix = torch.tril(torch.ones((x.size(1), x.size(1))))
          attention_out, _ = self.attention(x, x, x, mask_matrix)  # Using same input for query, key, value
        else:
          attention_out, _ = self.attention(x, x, x)  # Using same input for query, key, value

        # the output of ffn -> (batch_size, sequence_length, d_model)
        output = self.feed_forward(attention_out)
        # reduce the dimensions to 1 -> (batch_size, sequence_length, 1)
        output = self.linear1(output)
        # remove the last dimension -> (batch_size, sequence_length)
        output = output.squeeze(-1)
        # reduce the last dimension -> -> (batch_size, 1)
        output = self.linear2(output)

        return output

###########################################################
### Pipeline
###########################################################
def pipeline(
        X, Y, split_ratio,
        batch_size, sequence_length, d_model, 
        d_ffn, n_heads, dropout, 
        learning_rate, epochs
):  
    # build dataloader
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio=split_ratio, shuffle=shuffle)
    # model train
    model = TimeSeriesAttention(
        sequence_length, d_model, d_ffn, n_heads,
        dropout=dropout, bias_att=False
    )

    initial_weights = {name: param.clone() for name, param in model.state_dict().items() if "weight" or "bias" in name}
    print(initial_weights)

    model_trained, epoch_losses = train_model(model, dataloader_train, learning_rate, epochs)

    weights = output_weights(model_trained)
    print(weights)

    y_pred, mse = evaluate_model(model_trained, dataloader_test)

    return model_trained, epoch_losses, y_pred, mse
    


if __name__ == "__main__":

    sequence_length = 1 # number of tokens per input sequence
    d_model = 1 # number of dimension for each token
    split_ratio = 0.8

    n_heads = 1
    d_ffn = 1
    dropout = 0.0

    #shuffle = True
    shuffle = False
    learning_rate = 0.015
    epochs = 200
    #batch_sizes = [1, 2, 8, 128, 256, 512, 799]
    batch_sizes = [254, 255, 256, 257, 258]

    # reshape data for the model input
    X, Y = reshape_input(y, n, sequence_length, d_model)

    # model train
    all_epoch_losses = {}
    all_predictions = {}
    all_mses = {}

    # loop over different batch sizes
    for batch_size in batch_sizes:

        #seed = 42
        #seed = 123
        #seed = 2024
        #np.random.seed(seed)
        #torch.manual_seed(seed)
        print(f"Training with batch size: {batch_size}")
        print(f'seed = {seed}.')

        model_trained, epoch_losses, y_pred, mse = pipeline(X, Y, split_ratio, batch_size, sequence_length, d_model, d_ffn, n_heads, dropout, learning_rate, epochs)

        all_epoch_losses[batch_size] = epoch_losses
        all_predictions[batch_size] = y_pred
        all_mses[batch_size] = mse
        

    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))[::-1]

    # plot the losses for all batch sizes
    plt.figure(figsize=(10, 6))
    for i, (batch_size, losses) in enumerate(all_epoch_losses.items()):
        #plt.plot(np.log(losses), label=f'Batch size {batch_size}', color=colors[i])
        plt.plot(losses, label=f'Batch size {batch_size}', color=colors[i])

    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title(f'learning rate = {learning_rate}, shuffle = {shuffle}, seed = {seed}')
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f'losses_comparison_lr={learning_rate}_shuffl={shuffle}_seed={seed}.png')
    plt.show()

    # plot the predictions for all batch sizes
    plt.figure(figsize=(18, 4))
    split_index = int(split_ratio * n)
    plt.plot(np.arange(split_index, n), y[split_index:], label='True Data', color='red', alpha=0.5)
    for i, (batch_size, y_pred) in enumerate(all_predictions.items()):
        plt.plot(range(split_index, split_index + len(y_pred)), y_pred, label=f'Batch size {batch_size}', color=colors[i])
    
    plt.xlabel('Time steps')
    plt.title(f'learning rate = {learning_rate}, shuffle = {shuffle}, seed = {seed}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'predictions_comparison_lr={learning_rate}_shuffl={shuffle}_seed={seed}.png')
    plt.show()

    print(all_mses)

    ######################################
    ### Estimation with AR(1)
    ######################################
    # split data into training (80%) and testing (20%)
    split_index = int(0.8 * n)
    train_data = y[:split_index]
    test_data = y[split_index:]

    # fit AR(1) model using statsmodels
    ar_model = sm.tsa.ARIMA(train_data, order=(1, 0, 0))  # AR(1) is ARIMA with order (1,0,0)
    ar_model_fit = ar_model.fit()
    print(ar_model_fit.summary())
    # get the estimated parameter
    phi_estimated = ar_model_fit.params[1]  # AR coefficient
    intercept_estimated = ar_model_fit.params[0]

    # one-step-ahead prediction
    predictions = []
    last_observed_value = train_data[-1]  # Last value from training data

    for t in range(len(test_data)):
        next_prediction = intercept_estimated + phi_estimated * last_observed_value
        predictions.append(next_prediction)
        last_observed_value = test_data[t]

    # use the fitted AR(1) model to predict one-step ahead
    #predictions = ar_model_fit.predict(start=split_index, end=n-1, dynamic=True)
    # evaluate model performance using MSE
    mse = np.mean((test_data - predictions) ** 2)
    print(f"One-Step-Ahead Prediction MSE: {mse:.4f}")