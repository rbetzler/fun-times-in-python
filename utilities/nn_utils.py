"""base nn class"""
import abc
import math
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
from typing import Tuple


def _configure_torch(
    seed,
    deterministic,
    benchmark,
):
    """Set background pytorch configs"""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

def pad_data(
        batch_size: int,
        x: pd.DataFrame,
        y: pd.Series or pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Pad pandas df with zeros to conform to batch size"""
    n = len(x)
    print(f'Original df: {n}')
    while n % batch_size > 0:
        n += 1
    padding = n - len(x)
    if padding > 0:
        x_zeros = pd.DataFrame(np.zeros(shape=(padding, x.shape[1])), columns=x.columns)
        if isinstance(y, pd.Series):
            y_zeros = pd.Series(np.zeros(padding), name=y.name)
        else:
            y_zeros = pd.DataFrame(np.zeros(shape=(padding, y.shape[1])), columns=y.columns)
        x = x.append(x_zeros, ignore_index=True)
        y = y.append(y_zeros, ignore_index=True)
    print(f'Padded df: {len(x)}')
    return x, y

def determine_shapes(
        x,
        y,
        batch_size,
        sequence_length,
):
    """
    n_batches: actual number of batches to adjust for padding and seq length
    input_shape: (seq_length, batch_size, n_columns) of input
    output_shape: (n_rows, n_columns) of target
    """
    adj_batch_size = math.ceil(batch_size / sequence_length)
    n_batches = int(len(x) / adj_batch_size / sequence_length)
    n_columns = x.shape[1]
    input_shape = (
        sequence_length,
        adj_batch_size,
        n_columns
    )
    n_output_columns = 1 if isinstance(y, pd.Series) else y.shape[1]
    output_shape = sequence_length * adj_batch_size, n_output_columns
    return n_batches, input_shape, output_shape

def batch_data(
    x,
    y,
    n_batches,
):
    """Split x and y into batches"""
    x_arrays = np.array_split(x, n_batches)
    y_arrays = np.array_split(y, n_batches)
    data = []
    for n in range(0, n_batches):
        data.append((x_arrays[n], y_arrays[n]))
    return data

def fit(
    model,
    optimizer,
    loss_function,
    data,
    input_shape,
    n_epochs=100,
    device='cuda',
):
    """Fit pytorch model to training data"""
    batch = 0
    for d in data:
        batch += 1
        x = torch.tensor(d[0].values).to(device).float().view(input_shape)
        y = torch.tensor(d[1].values).to(device).float()

        for epoch in range(n_epochs):
            prediction = model.forward(x)

            loss = loss_function(prediction.view(-1), y.view(-1))
            if epoch % int(n_epochs/10) == 0:
                print(f'Batch {batch}, Epoch {epoch}, Loss {loss.item()}')

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

def predict(
    model,
    x,
    n_batches,
    input_shape,
    output_shape,
    device='cuda',
):
    """Predict on input data"""
    model.eval()
    predictions = np.empty((0, output_shape[1]))
    arrays = np.array_split(x, n_batches)
    for array in arrays:
        x_out = torch.tensor(array.values).float().view(input_shape).to(device)
        prediction = model.forward(x_out)
        predictions = np.append(predictions, prediction.view(output_shape).cpu().detach().numpy(), axis=0)
    return predictions
