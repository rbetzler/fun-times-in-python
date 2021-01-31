import pandas as pd
import torch
import unittest
from science.models import lstm

X0 = 'x0'
X1 = 'x1'
X2 = 'x2'
Y = 'y'


class LSTM0(unittest.TestCase):
    """Unit tests for LSTM0"""

    @property
    def x(self):
        df = pd.DataFrame(
            data=[
                [.7, .42, .72],
                [.62, .56, .44],
                [.83, .63, .9],
                [.31, .22, .43],
            ],
            columns=[X0, X1, X2],
        )
        return df

    @property
    def y(self):
        df = pd.DataFrame(
            data=[.4, .5, .6, .4],
            columns=[Y],
        )
        return df

    def test_training(self):
        """
        Whether lstm training updates parameters
        """
        for n_layers in [1, 3]:
            model = lstm.LSTM0(
                x=self.x,
                y=self.y,
                batch_size=len(self.x),
                n_layers=n_layers,
            )
            initial_parameters = [(p[0], p[1].clone()) for p in model.named_parameters() if p[1].requires_grad]
            model.fit()
            trained_parameters = [(p[0], p[1].clone()) for p in model.named_parameters() if p[1].requires_grad]
            for n in range(0, len(initial_parameters)):
                assert not torch.equal(initial_parameters[n][1], trained_parameters[n][1]), 'Parameters did not change!'


if __name__ == '__main__':
    unittest.main()
