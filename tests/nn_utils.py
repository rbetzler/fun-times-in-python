import pandas as pd
import torch
import unittest

from pandas import testing as pdt
from utilities import nn_utils

X0 = 'x0'
X1 = 'x1'
X2 = 'x2'
Y = 'y'


class SimpleNN(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape=1,
    ):
        super(SimpleNN, self).__init__()
        self.linear0 = torch.nn.Linear(
            input_shape,
            output_shape,
        )

    def forward(self, x):
        output = self.linear0(x)
        return output


class NNUtils(unittest.TestCase):
    """Unit tests for the Base LSTM class"""

    @property
    def x(self):
        df = pd.DataFrame(
            data=[
                [10, 40, 70],
                [20, 50, 80],
                [30, 60, 90],
                [11, 22, 33],
            ],
            columns=[X0, X1, X2],
        )
        return df

    @property
    def y(self):
        df = pd.DataFrame(
            data=[1, 2, 3, 4],
            columns=[Y],
        )
        return df

    @property
    def y_series(self):
        s = pd.Series(
            data=[1, 2, 3, 4],
            name=Y,
        )
        return s

    @staticmethod
    def expectation_pad_data():
        x = pd.DataFrame(
            data=[
                [10, 40, 70],
                [20, 50, 80],
                [30, 60, 90],
                [11, 22, 33],
                [0, 0, 0],
                [0, 0, 0],
            ],
            columns=[X0, X1, X2],
            dtype=float,
        )
        y = pd.Series(
            data=[1, 2, 3, 4, 0, 0],
            name=Y,
            dtype=float,
        )
        return x, y

    def test_pad_data(self):
        """
        Whether dataframe padding works correctly
        """
        x, y = nn_utils.pad_data(
           batch_size=3,
           x=self.x,
           y=self.y_series,
        )
        exp_x, exp_y = self.expectation_pad_data()
        pdt.assert_frame_equal(exp_x, x)
        pdt.assert_series_equal(exp_y, y)

    def test_determine_shapes(self):
        n_batches, input_shape, output_shape = nn_utils.determine_shapes(
            x=self.x,
            y=self.y,
            batch_size=4,
            sequence_length=2,
        )
        self.assertEqual(1, n_batches)
        self.assertEqual((2, 2, 3), input_shape)
        self.assertEqual((4, 1), output_shape)

    @property
    def expectation_batch_data(self):
        one = self.x.head(2), self.y.head(2)
        two = self.x.tail(2), self.y.tail(2)
        return [one, two]

    def test_batch_data(self):
        data = nn_utils.batch_data(
            x=self.x,
            y=self.y,
            n_batches=2,
        )
        pdt.assert_frame_equal(self.expectation_batch_data[0][0], data[0][0])
        pdt.assert_frame_equal(self.expectation_batch_data[0][1], data[0][1])
        pdt.assert_frame_equal(self.expectation_batch_data[1][0], data[1][0])
        pdt.assert_frame_equal(self.expectation_batch_data[1][1], data[1][1])

    def test_fit(self):
        model = SimpleNN(input_shape=self.x.shape[1]).to('cuda')
        initial_parameters = [(p[0], p[1].clone()) for p in model.named_parameters() if p[1].requires_grad]
        nn_utils.fit(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=.01),
            loss_function=torch.nn.L1Loss(reduction='sum'),
            data=self.expectation_batch_data,
            input_shape=(1, 2, 3),
            n_epochs=10,
        )
        trained_parameters = [(p[0], p[1].clone()) for p in model.named_parameters() if p[1].requires_grad]
        for n in range(0, len(initial_parameters)):
            assert not torch.equal(initial_parameters[n][1], trained_parameters[n][1]), 'Parameters did not change!'


if __name__ == '__main__':
    unittest.main()
