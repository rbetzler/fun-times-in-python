"""
python science/executor.py --job=s1 --start_date='2016-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s1 --start_date='2019-02-01' --n_days=30 -ab
"""
from science.predictor import thirty_day_low as tdl


class S1(tdl.ThirtyDayLowPredictorNN):
    @property
    def model_id(self) -> str:
        return 's1'

    @property
    def hidden_shape(self) -> int:
        return 1000
