import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.attention import MultiHeadAttention, MultiHeadAttention2
from models.feed_forward import PositionwiseFeedForward
from models.encoder import Encoder
from models.decoder import Decoder

from train_and_test.train_and_test import train_model, evaluate_model, QuantileLoss
from utils.utils import output_weights, reshape_input, split_and_build_dataloader
from utils.visualize import plot_predictions, plot_loss

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

###########################################################
### Data Generation Process
###########################################################

n = 1000  # number of data points
phi = 0.4
sigma = 0.2  # sd. of noise
tau = 0.25   

y = np.zeros(n)
epsilon = np.random.normal(scale=sigma, size=n) - sigma * tau
y[0] = epsilon[0] # initialize the first value of the series
for t in range(1, n):
    y[t] = phi * y[t-1] + epsilon[t]

# VaR calculation: estimate the VaR based on the historical data (5th percentile)
Q_tau = np.percentile(y, 100 * tau)

VaR_t = np.zeros(n)
for t in range(1, n):
    # Conditional quantile for each time step based on past information
    conditional_mean = phi * y[t - 1]  # mean of AR(1) at time t
    VaR_t[t] = conditional_mean + np.percentile(epsilon[:t], tau * 100)  # quantile of residuals



plt.figure(figsize=(10, 6))
plt.plot(y, label='AR(1) Time Series', color='blue', alpha=0.7)
plt.plot(VaR_t, label=f'VaR at tau={tau}', color='red', linestyle='--')
plt.title(f'Time-Varying VaR at tau = {tau} for AR(1) Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

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
        output = self.relu(attention_out + 0.3)
        output = self.feed_forward(attention_out)
        # reduce the dimensions to 1 -> (batch_size, sequence_length, 1)
        output = self.linear1(output)
        # remove the last dimension -> (batch_size, sequence_length)
        output = output.squeeze(-1)
        # reduce the last dimension -> -> (batch_size, 1)
        output = self.linear2(output)

        return output
    
if __name__ == "__main__":

    sequence_length = 1 # number of tokens per input sequence
    d_model = 1 # number of dimension for each token
    batch_size = 8
    split_ratio = 0.8

    tau = 0.2
    n_heads = 1
    d_ffn = 32
    dropout = 0.0
    learning_rate = 0.0015
    epochs = 20

    # reshape data for the model input
    X, Y = reshape_input(y, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    # model train
    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=False
    )
    q_loss = QuantileLoss(tau)
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs, criterion=q_loss)
    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_quantile.png")

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    plot_predictions(y_pred=y_pred, y=y, VaR=Q_tau, Var_t=VaR_t, split_ratio=split_ratio, filename="figs/pred_quantile.png", mse=mse)