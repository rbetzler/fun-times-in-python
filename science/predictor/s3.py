"""
python science/executor.py --job=s3 --start_date='2016-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s3 --start_date='2019-02-01' --n_days=30 -ab
"""
from science.predictor import thirty_day_low as tdl


class S3(tdl.ThirtyDayLowPredictorLSTM):
    @property
    def model_id(self) -> str:
        return 's3'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 800,
            'hidden_shape': 750,
            'dropout': 0.2,
            'learning_rate': .0001,
            'seed': 33,
            'sequence_length': 10,
            'batch_size': 31000,
        }
        return kwargs
