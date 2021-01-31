"""lstm utils"""
import torch
from science.models import base


class LSTM0(base.LSTM):

    def _configure_network(self):
        self.lstm = torch.nn.LSTM(
            input_size=self.input_shape,
            hidden_size=self.hidden_shape,
            num_layers=self.n_layers,
            bias=self.bias,
            dropout=self.dropout,
        ).to(self.device)
        self.relu = torch.nn.ReLU6()
        self.linear = torch.nn.Linear(self.hidden_shape, self.output_shape).to(self.device)

        # TODO: Get parallelization working with hidden states
        # self.lstm = torch.nn.DataParallel(self.lstm)
        # self.relu = torch.nn.DataParallel(self.relu)
        # self.linear = torch.nn.DataParallel(self.linear)

    @property
    def loss_function(self):
        loss = torch.nn.L1Loss(reduction='sum').to(self.device)
        return loss

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, data):
        # self.lstm.flatten_parameters()
        # output, self.hidden = self.lstm(data, self.hidden)
        output, _ = self.lstm(data)
        output = self.linear(output)
        output = self.relu(output)
        return output
