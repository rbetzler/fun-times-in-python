"""
python science/executor.py --job=a5 --start_date='2019-01-15' --n_days=200 --is_training_run
python science/executor.py --job=a5 --start_date='2020-08-03' --n_days=30 -ab
"""
from science.predictor import stock_autoencoder as sae


class A5(sae.StockAutoencoder):
    @property
    def model_id(self) -> str:
        return 'a5'
