"""autoencoder implementation 0"""
import torch


class Autoencoder0(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_shape,
    ):
        super(Autoencoder0, self).__init__()
        self.linear0 = torch.nn.Linear(
            input_shape,
            hidden_shape,
        )
        self.relu = torch.nn.ReLU6()
        self.linear1 = torch.nn.Linear(
            hidden_shape,
            input_shape,
        )

    def forward(self, x):
        output = self.linear0(x)
        output = self.relu(output)
        output = self.linear1(output)
        return output
