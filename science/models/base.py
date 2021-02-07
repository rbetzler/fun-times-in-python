"""base nn class"""
import abc
import math
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
from typing import Tuple


class NN(torch.nn.Module):
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
            device: int = 0,
            bias: bool = True,
            dropout: float = 0,
            seed: int = 3,
            deterministic: bool = True,
            benchmark: bool = False,
            pad: bool = True,
    ):
        super(NN, self).__init__()
        self._configure_torch(
            seed=seed,
            deterministic=deterministic,
            benchmark=benchmark,
        )
        self._set_params(
            n_layers=n_layers,
            hidden_shape=hidden_shape,
            output_shape=output_shape,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device=device,
            bias=bias,
            dropout=dropout,
        )
        self._arrange_data(
            x=x,
            y=y,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        print(f'''
        NN Config
            Sequence Length: {sequence_length}
            Batch Size: {batch_size}
            Adj. Batch Size: {self.batch_size}
            Dataset: {len(x)}
            Padded Dataset: {len(self.x)}
            N Training Batches: {self.n_training_batches}
            Device: {self.device}
        ''')
        self.hidden = None
        self._configure_network()

    @staticmethod
    def _configure_torch(seed, deterministic, benchmark):
        """Set some background pytorch configs"""
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

    def _set_params(
            self,
            n_layers,
            hidden_shape,
            output_shape,
            n_epochs,
            learning_rate,
            device,
            bias,
            dropout,
    ):
        """Set (most) init params"""
        self.n_layers = n_layers
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = f'cuda:{device}'
        self.bias = bias
        self.dropout = dropout

    def _arrange_data(
            self,
            x,
            y,
            batch_size,
            sequence_length,
    ):
        """Conform data and set as attribute"""
        self.input_shape = x.shape[1]
        self.batch_size = math.ceil(batch_size / sequence_length)
        self.x, self.y = self.pad(batch_size, x, y)
        self.n_training_batches = int(len(self.x) / self.batch_size / sequence_length)
        self.input = (
            sequence_length,
            self.batch_size,
            self.input_shape
        )

    @abc.abstractmethod
    def _configure_network(self):
        """Network structure"""
        pass

    @staticmethod
    def split(df, n):
        """Split pandas data frame by batch size"""
        return np.array_split(df, n)

    @staticmethod
    def pad(
            batch_size: int,
            x: pd.DataFrame,
            y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
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
        x = self.split(self.x, self.n_training_batches)
        y = self.split(self.y, self.n_training_batches)
        data = []
        for n in range(0, self.n_training_batches):
            data.append((x[n], y[n]))
        return data

    @property
    @abc.abstractmethod
    def loss_function(self):
        """Loss function to use when training"""
        pass

    @property
    @abc.abstractmethod
    def optimizer(self):
        """Optimizer for training"""
        pass

    @abc.abstractmethod
    def forward(self, data):
        """Forward pass to make when training or testing"""
        pass

    def fit(self):
        """Fix pytorch model on training data"""
        optimizer = self.optimizer
        batch = 0
        for data in self.training_data():
            batch += 1
            x = torch.tensor(data[0].values).to(self.device).float().view(self.input)
            y = torch.tensor(data[1].values).to(self.device).float()

            for epoch in range(self.n_epochs):
                prediction = self.forward(x)

                loss = self.loss_function(prediction.view(-1), y.view(-1))
                if epoch % int(self.n_epochs/10) == 0:
                    print(f'Batch {batch}, Epoch {epoch}, Loss {loss.item()}')

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

    def predict(self):
        """Predict on input data"""
        self.eval()
        predictions = np.array([])
        arrays = self.split(self.x, self.n_training_batches)
        for array in arrays:
            x = torch.tensor(array.values).to(self.device).float().view(self.input)
            prediction = self.forward(x)
            predictions = np.append(predictions, prediction.view(-1).cpu().detach().numpy())
        return predictions

    @property
    def prediction_df(self):
        """Convert model prediction to user friendly dataframe"""
        df = self.x.copy()
        df['prediction'] = self.predict()
        return df
