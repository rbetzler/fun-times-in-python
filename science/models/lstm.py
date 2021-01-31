"""lstm utils"""
from torch import nn, optim
from science.models import base


class LSTM0(base.LSTM):

    def _configure_network(self):
        # Network
        self.lstm = nn.LSTM(
            input_size=self.input_shape,
            hidden_size=self.hidden_shape,
            num_layers=self.n_layers,
            bias=self.bias,
            dropout=self.dropout,
        ).to(self.device)

        self.relu = nn.ReLU6()
        self.linear = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)

        # Parallelization
        self.lstm = nn.DataParallel(self.lstm)
        self.relu = nn.DataParallel(self.relu)
        self.linear = nn.DataParallel(self.linear)

    @property
    def loss_function(self):
        loss = nn.L1Loss(reduction='sum').to(self.device)
        return loss

    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, data):
        self.lstm.module.flatten_parameters()
        output, _ = self.lstm(data, None)
        output = self.linear(output)
        output = self.relu(output)
        return output
