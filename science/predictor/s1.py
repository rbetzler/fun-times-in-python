"""
python science/executor.py --job=s1 --start_date='2015-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s1 --start_date='2019-02-01' --n_days=30 -ab
"""
import abc
import pandas as pd

from science.predictor import base


class ThirtyDayLowPredictor(base.Predictor, abc.ABC):
    """Predict the lowest stock prices over the next 30 days"""

    @property
    def query(self) -> str:
        query = f'''
            select
                symbol
              , market_datetime
              , target
              , denormalized_target
              , normalization_min
              , normalization_max
              , open_1
              , open_2
              , open_3
              , open_4
              , open_5
              , open_6
              , open_7
              , open_8
              , open_9
              , open_10
              , open_11
              , open_12
              , open_13
              , open_14
              , open_15
              , open_16
              , open_17
              , open_18
              , open_19
              , open_20
              , open_21
              , open_22
              , open_23
              , open_24
              , open_25
              , open_26
              , open_27
              , open_28
              , open_29
              , open_30
              , mean_deviation_10_over_30
              , mean_deviation_10_over_60
              , mean_deviation_10_over_90
              , max_deviation_10_over_30
              , max_deviation_10_over_60
              , max_deviation_10_over_90
            from dbt.training
            where market_datetime between '{self.start_date}' and '{self.end_date}'
              {'and target is not null' if self.is_training_run else ''}
            order by market_datetime, symbol
            { 'limit ' + str(self.limit) if self.is_training_run else '' }
            { 'offset ' + str(self.limit * self.n_subrun) if self.is_training_run else '' }
            '''
        return query


class S1(ThirtyDayLowPredictor):
    @property
    def model_id(self) -> str:
        return 's1'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 200,
            'hidden_shape': 1000,
            'dropout': 0.1,
            'learning_rate': .0001,
            'seed': 55,
            'sequence_length': 2,
            'batch_size': 31000,
        }
        return kwargs
