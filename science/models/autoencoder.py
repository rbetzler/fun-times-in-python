"""nn implementation of an autoencoder"""
import pandas as pd
import torch
from science.models import base


class NNAutoencoder0(base.NN):

    def _configure_network(self):
        self.linear0 = torch.nn.Linear(
            self.input_shape,
            self.hidden_shape,
        ).to(self.device)
        self.linear1 = torch.nn.Linear(
            self.hidden_shape,
            self.input_shape,
        ).to(self.device)

    @property
    def loss_function(self):
        loss = torch.nn.L1Loss(reduction='sum').to(self.device)
        return loss

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, data):
        output = self.linear0(data)
        output = self.linear1(output)
        return output

    @property
    def prediction_df(self):
        predictions = self.predict()
        df = pd.DataFrame(predictions, columns=self.x.columns)
        return df
