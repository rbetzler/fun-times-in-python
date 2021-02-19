"""
python science/executor.py --job=s4 --start_date='2019-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s4 --start_date='2020-08-03' --n_days=30 -ab
"""
from science.predictor import thirty_day_low as tdl


class S5(tdl.ThirtyDayLowPredictorNN):
    @property
    def model_id(self) -> str:
        return 's5'

    @property
    def batch_size(self) -> int:
        return 8000

    @property
    def sequence_length(self) -> int:
        return 2
