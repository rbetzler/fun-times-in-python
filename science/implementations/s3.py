"""
python science/executor.py --job=s3 --start_date='2019-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s3 --start_date='2020-08-03' --n_days=30 -ab
"""
from science.predictors import speculation as spc


class S3(spc.LowSpeculativePredictorNN):
    @property
    def model_id(self) -> str:
        return 's3'

    @property
    def hidden_shape(self) -> int:
        return 500

    @property
    def batch_size(self) -> int:
        return 4000

    @property
    def limit(self) -> int:
        """When backtesting, the size of the dataset"""
        return 12000
