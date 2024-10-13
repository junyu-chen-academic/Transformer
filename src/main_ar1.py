import time
import numpy as np

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

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

###########################################################
### Data Generation Process
###########################################################
phi = 0.4
sigma = 0.2  # sd. of noise
n = 1000  # number of time points

y_ar1 = np.zeros(n)
epsilon = np.random.normal(0, sigma, n)
y_ar1[0] = epsilon[0] # initialize the first value of the series

# generate AR(1) time series
for t in range(1, n):
    y_ar1[t] = phi * y_ar1[t-1] + epsilon[t]
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
    
if __name__ == "__main__":

    sequence_length = 1 # number of tokens per input sequence
    batch_size = 8
    split_ratio = 0.8
    dropout = 0.0
    learning_rate = 0.0015
    epochs = 40


    ################################################################################
    # model train
    d_model = 2
    n_heads = 2
    d_ffn = 1
    bias_att = False
    filename = f"d={d_model}_h={n_heads}_d_ffn={d_ffn}_{bias_att}_att_bias.png"

    # reshape data for the model input
    X, Y = reshape_input(y_ar1, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=bias_att
    )

    start_time = time.time()
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_"+filename)

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    # plot the
    plot_predictions(y_pred=y_pred, y=y_ar1, split_ratio=split_ratio, filename="figs/pred_"+filename, mse=mse)
    ################################################################################
 
    ################################################################################
    # model train
    d_model = 4
    n_heads = 2
    d_ffn = 1
    bias_att = False
    filename = f"d={d_model}_h={n_heads}_d_ffn={d_ffn}_{bias_att}_att_bias.png"

    # reshape data for the model input
    X, Y = reshape_input(y_ar1, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=bias_att
    )

    start_time = time.time()
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_"+filename)

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    # plot the
    plot_predictions(y_pred=y_pred, y=y_ar1, split_ratio=split_ratio, filename="figs/pred_"+filename, mse=mse)
    ################################################################################

    ################################################################################
    # model train
    d_model = 2
    n_heads = 2
    d_ffn = 2
    bias_att = False
    filename = f"d={d_model}_h={n_heads}_d_ffn={d_ffn}_{bias_att}_att_bias.png"

    # reshape data for the model input
    X, Y = reshape_input(y_ar1, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=bias_att
    )

    start_time = time.time()
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_"+filename)

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    # plot the
    plot_predictions(y_pred=y_pred, y=y_ar1, split_ratio=split_ratio, filename="figs/pred_"+filename, mse=mse)
    ################################################################################

    ################################################################################
    # model train
    d_model = 4
    n_heads = 2
    d_ffn = 2
    bias_att = False
    filename = f"d={d_model}_h={n_heads}_d_ffn={d_ffn}_{bias_att}_att_bias.png"

    # reshape data for the model input
    X, Y = reshape_input(y_ar1, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=bias_att
    )

    start_time = time.time()
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_"+filename)

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    # plot the
    plot_predictions(y_pred=y_pred, y=y_ar1, split_ratio=split_ratio, filename="figs/pred_"+filename, mse=mse)
    ################################################################################

    ################################################################################
    # model train
    d_model = 2
    n_heads = 2
    d_ffn = 8
    bias_att = False
    filename = f"d={d_model}_h={n_heads}_d_ffn={d_ffn}_{bias_att}_att_bias.png"

    # reshape data for the model input
    X, Y = reshape_input(y_ar1, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=bias_att
    )

    start_time = time.time()
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_"+filename)

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    # plot the
    plot_predictions(y_pred=y_pred, y=y_ar1, split_ratio=split_ratio, filename="figs/pred_"+filename, mse=mse)
    ################################################################################

    ################################################################################
    # model train
    d_model = 4
    n_heads = 2
    d_ffn = 8
    bias_att = False
    filename = f"d={d_model}_h={n_heads}_d_ffn={d_ffn}_{bias_att}_att_bias.png"

    # reshape data for the model input
    X, Y = reshape_input(y_ar1, n, sequence_length, d_model)
    dataloader_train, dataloader_test = split_and_build_dataloader(X, Y, batch_size=batch_size, split_ratio = split_ratio)

    model1 = TimeSeriesAttention(
        sequence_length=sequence_length, d_model=d_model, d_ffn=d_ffn, n_heads=n_heads,
        dropout=dropout, bias_att=bias_att
    )

    start_time = time.time()
    model_trained, epoch_losses = train_model(model1, dataloader_train, learning_rate, epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    weights = output_weights(model_trained)
    plot_loss(epoch_losses, filename="figs/loss_"+filename)

    # evaluate the model
    y_pred, mse = evaluate_model(model1, dataloader_test)
    # plot the
    plot_predictions(y_pred=y_pred, y=y_ar1, split_ratio=split_ratio, filename="figs/pred_"+filename, mse=mse)
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################



    