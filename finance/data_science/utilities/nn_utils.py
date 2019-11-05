import pandas as pd
import torch
from torch import nn, optim


class TorchNN(nn.Module):
    def __init__(self,
                 train_x=pd.DataFrame,
                 train_y=pd.DataFrame,
                 hidden_size=8,
                 output_size=1,
                 learning_rate=0,
                 momentum=.9):

        self.train_x = torch.tensor(train_x.values).float().view(1, -1).detach().requires_grad_(True)
        self.train_y = torch.tensor(train_y.values).float()

        self.size = train_x.shape[0] * train_x.shape[1]
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.momentum = momentum

    @property
    def layers(self):
        layers = {
            'linear_one': nn.Linear(in_features=self.size, out_features=self.hidden_size).float(),
            'linear_two': nn.Linear(in_features=self.hidden_size, out_features=self.output_size).float()
        }
        return layers

    def forward(self, data):
        for layer in self.layers:
            data = self.layers.get(layer)(data)
        return data

    def optimizer(self):
        return optim.SGD([self.train_x], lr=self.learning_rate, momentum=self.momentum)

    @property
    def criterion(self):
        return nn.MSELoss()

    @property
    def n_epochs(self):
        return 2

    def train_network(self):
        running_loss = 0
        optimizer = self.optimizer()
        criterion = self.criterion

        for epoch in range(self.n_epochs):
            for data, target in zip(self.train_x, self.train_y):
                optimizer.zero_grad()

                outputs = self.forward(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
