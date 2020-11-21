"""lstm utils"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

import torch
from torch import nn, optim


class TorchLSTM(nn.Module):
    def __init__(
            self,
            x: pd.DataFrame = None,
            y: pd.DataFrame = None,
            n_layers=2,
            hidden_shape=100,
            output_shape=1,
            sequence_length=1,
            batch_size=None,
            n_epochs: int = 100,
            learning_rate: float = .0001,
            device: str = 'cuda',
            bias: bool = True,
            dropout: float = 0,
            seed: int = 3,
            deterministic: bool = True,
            benchmark: bool = False,
            pad: bool = True,
    ):

        super(TorchLSTM, self).__init__()

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

        # Network params
        self.n_layers = n_layers
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        # Learning params
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Data and dimensions
        self.input_shape = x.shape[1]

        self.batch_size = int(batch_size / sequence_length)
        self.x, self.y = self.pad(self.batch_size, x, y)
        self.n_training_batches = max(int(len(self.x) / batch_size), 1)

        self.input = (
            sequence_length,
            self.batch_size,
            self.input_shape
        )

        # Network
        self.lstm = nn.LSTM(
            input_size=self.input_shape,
            hidden_size=self.hidden_shape,
            num_layers=self.n_layers,
            bias=bias,
            dropout=dropout,
        ).to(self.device)

        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)

        # Parallelization
        self.lstm = nn.DataParallel(self.lstm)
        self.relu = nn.DataParallel(self.relu)
        self.linear = nn.DataParallel(self.linear)

    def reset_network(self):
        self.lstm.reset_parameters()
        self.linear.reset_parameters()

    @staticmethod
    def pad(
            batch_size: int,
            x: pd.DataFrame,
            y: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pad raw pandas df with zeros"""
        n = len(x)
        while n % batch_size > 0:
            n += 1
        padding = n - len(x)
        if padding > 0:
            x_zeros = pd.DataFrame(np.zeros(shape=(padding, x.shape[1])), columns=x.columns)
            y_zeros = pd.Series(np.zeros(padding), name=y.name)
            x = x.append(x_zeros, ignore_index=True)
            y = y.append(y_zeros, ignore_index=True)
        return x, y

    def training_data(self):
        x = np.array_split(self.x, self.n_training_batches)
        y = np.array_split(self.y, self.n_training_batches)

        data = []
        for n in range(0, self.n_training_batches):
            data.append((x[n], y[n]))
        return data

    @property
    def loss_function(self):
        loss = nn.L1Loss(reduction='sum').to(self.device)
        return loss

    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # def create_hidden_states(self):
    #     hidden = torch.zeros(self.n_layers,
    #                          self.batch_size,
    #                          self.hidden_shape).to(self.device)
    #
    #     cell = torch.zeros(self.n_layers,
    #                        self.batch_size,
    #                        self.hidden_shape).to(self.device)
    #     return hidden, cell

    def forward(self, data):
        # TODO: figure out what hidden states actually do
        # output, self.hidden = self.lstm(data, self.hidden)

        self.lstm.module.flatten_parameters()
        output, _ = self.lstm(data, None)
        output = self.relu(output)
        output = self.linear(output)
        return output

    def fit(self):
        optimizer = self.optimizer
        history = []

        batch = 0
        for data in self.training_data():
            batch += 1
            x = torch.tensor(data[0].values).to(self.device).float().detach().requires_grad_(True).view(self.input)
            y = torch.tensor(data[1].values).to(self.device).float()

            for epoch in range(self.n_epochs):
                prediction = self.forward(x)

                loss = self.loss_function(prediction.view(-1), y.view(-1))
                if epoch % int(self.n_epochs/10) == 0:
                    print(f'Batch {batch}, Epoch {epoch}, Loss {loss.item()}')

                history.append([batch, epoch, loss.item()])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        df_history = pd.DataFrame(history, columns=['batch', 'epoch', 'loss'])

        plt.title('Cumulative Loss by Epoch')
        plt.plot(df_history.groupby('epoch').sum()['loss'])

    def predict(self):
        self.eval()
        predictions = np.array([])
        arrays = np.array_split(self.x, self.n_training_batches)
        for array in arrays:
            x = torch.tensor(array.values).to(self.device).float().detach().view(self.input)
            prediction = self.forward(x)
            predictions = np.append(predictions, prediction.view(-1).cpu().detach().numpy())
        return predictions

    @property
    def prediction_df(self):
        df = self.x.copy()
        df['prediction'] = self.predict()
        return df

    # TODO: Add and refine basic performance features
