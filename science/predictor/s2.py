"""
python science/executor.py --job=s2 --start_date='2016-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s2 --start_date='2019-02-01' --n_days=30 -ab
"""
from science.predictor import thirty_day_low as tdl


class S2(tdl.ThirtyDayLowPredictor):
    @property
    def model_id(self) -> str:
        return 's2'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 500,
            'hidden_shape': 500,
            'dropout': 0.1,
            'learning_rate': .0001,
            'seed': 44,
            'sequence_length': 2,
            'batch_size': 31000,
        }
        return kwargs
