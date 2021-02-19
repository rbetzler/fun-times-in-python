"""
python science/executor.py --job=s2 --start_date='2019-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s2 --start_date='2020-08-03' --n_days=30 -ab
"""
from science.predictor import thirty_day_low as tdl


class S2(tdl.ThirtyDayLowPredictorLSTM):
    @property
    def model_id(self) -> str:
        return 's2'

    @property
    def hidden_shape(self) -> int:
        return 1000

    @property
    def batch_size(self) -> int:
        return 10000

    @property
    def limit(self) -> int:
        """When backtesting, the size of the dataset"""
        return 40000
