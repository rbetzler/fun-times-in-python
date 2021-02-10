"""
python science/executor.py --job=a5 --start_date='2019-01-15' --n_days=200 --is_training_run
python science/executor.py --job=a5 --start_date='2020-08-03' --n_days=30 -ab
"""
from science.predictor import stock_autoencoder as sae


class A5(sae.StockAutoencoder):
    @property
    def model_id(self) -> str:
        return 'a5'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 400,
            'hidden_shape': 30,
            'dropout': 0.2,
            'learning_rate': .0001,
            'seed': 55,
            'sequence_length': 10,
            'batch_size': 31000,
        }
        return kwargs
