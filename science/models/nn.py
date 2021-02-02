"""nn implementation 0"""
import torch
from science.models import base


class NN0(base.NN):

    def _configure_network(self):
        self.linear0 = torch.nn.Linear(
            self.input_shape,
            self.hidden_shape,
        ).to(self.device)
        self.relu = torch.nn.ReLU6()
        self.linear1 = torch.nn.Linear(
            self.hidden_shape,
            self.output_shape,
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
        output = self.relu(output)
        output = self.linear1(output)
        return output
