import pandas as pd
import unittest
from science.models import base
from pandas import testing as pdt

X0 = 'x0'
X1 = 'x1'
X2 = 'x2'
Y = 'y'


class BaseLSTM(unittest.TestCase):
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
    def expectation_split():
        x = pd.DataFrame(data=[1, 2], columns=[Y], index=[0, 1])
        y = pd.DataFrame(data=[3], columns=[Y], index=[2])
        z = pd.DataFrame(data=[4], columns=[Y], index=[3])
        return [x, y, z]

    def test_split(self):
        """
        Whether lstm dataframe splitting works correctly
        """
        case = base.LSTM.split(df=self.y, n=3)
        expectation = self.expectation_split()
        self.assertEqual(len(expectation), len(case))
        for n in range(0, len(case)):
            pdt.assert_frame_equal(expectation[n], case[n])

    @staticmethod
    def expectation_pad():
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

    def test_pad(self):
        """
        Whether lstm dataframe padding works correctly
        """
        x, y = base.LSTM.pad(
           batch_size=3,
           x=self.x,
           y=self.y_series,
        )
        exp_x, exp_y = self.expectation_pad()
        pdt.assert_frame_equal(exp_x, x)
        pdt.assert_series_equal(exp_y, y)


if __name__ == '__main__':
    unittest.main()
