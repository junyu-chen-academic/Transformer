import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

######################################
### Quantile Loss
######################################
class QuantileLoss(nn.Module):
	"""
	implements the quantile loss.
	"""
	def __init__(self, tau: float):
		super().__init__()
		self.tau = tau

	def forward(self, predicted, target):
		errors = target - predicted

		loss = torch.max(
			(self.tau - 1) * errors,
			self.tau * errors
		)

		return loss.mean()
	
######################################
### Model Training
######################################
def train_model(
		model: nn.Module,
		dataloader: DataLoader,
		learning_rate: float,
		epochs: int,
		criterion: nn.Module = None,
		mask_or_not: bool = False
):
	"""
    Trains the given model on data using an optimizer and loss function.

    Args:
        model:            The model to train.
        dataloader:       DataLoader to provide training data.
        learning_rate:    Learning rate for the optimizer.
        epochs:           Number of training epochs.
        criterion:        Loss function, defaults to MSELoss.
        mask_or_not:      Whether to apply a mask, defaults to False.

    Returns:
        nn.Module:        The trained model.
    """
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	if criterion is None:
		criterion = nn.MSELoss()
	
	epoch_losses = [] 
	
	for epoch in range(epochs):
		
		epoch_loss = 0

		for input, target in dataloader:

			optimizer.zero_grad()
			# forward pass
			predicted = model(input, mask_or_not=mask_or_not)

			# calculate loss
			loss = criterion(predicted, target)
			# backward pass
			loss.backward()
			# optimization
			optimizer.step()

			# Accumulate loss for the current epoch
			epoch_loss += loss.item()

		epoch_loss /= len(dataloader)
		epoch_losses.append(epoch_loss)
	
		print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
	
	return model, epoch_losses

######################################
### Model Evaluation
######################################
def evaluate_model(
		model: nn.Module,
		dataloader: DataLoader,
		mask_or_not: bool = False,
):
	"""
    Evaluates the given model on data.

    Args:
        model:            The model to evaluate.
        dataloader:       DataLoader to provide evaluation data.

    Returns:
        tuple: A tuple containing:
            - predictions (np.ndarray):   Predicted values.
            - mse (float):                The mean squared error (MSE) between the model's predictions and targets.
    """
	model.eval()

	# lists
	predictions = []
	targets = []

	with torch.no_grad():
		for input, target in dataloader:
			# perform forward pass
			predicted = model(input, mask_or_not=mask_or_not)

			# store predictions and actual values
			predictions.append(predicted.numpy())
			targets.append(target.numpy())

  	# convert the lists to numpy arrays
	predictions = np.concatenate(predictions)
	targets = np.concatenate(targets)

  	# calculate MSE for evaluation
	mse = np.mean((predictions - targets) ** 2)
	print(f"Mean Squared Error: {mse:.4f}")

	return predictions, mse