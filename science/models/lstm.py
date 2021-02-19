"""lstm implementation 0"""
import torch


class LSTM0(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_shape,
        output_shape,
        n_layers=2,
        bias=True,
        dropout=.15,
    ):
        super(LSTM0, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_shape,
            hidden_shape,
            num_layers=n_layers,
            bias=bias,
            dropout=dropout,
        )
        self.relu = torch.nn.ReLU6()
        self.linear = torch.nn.Linear(
            hidden_shape,
            output_shape,
        )

    def forward(self, x):
        # self.lstm.flatten_parameters()
        # output, self.hidden = self.lstm(data, self.hidden)
        output, _ = self.lstm(x)
        output = self.relu(output)
        output = self.linear(output)
        return output
