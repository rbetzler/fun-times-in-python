"""
python science/executor.py --job=s4 --start_date='2015-01-15' --n_days=200 --is_training_run
python science/executor.py --job=s4 --start_date='2018-01-19' --n_days=30 -ab
"""
from science.predictor import s1


class S4(s1.Predictor):
    @property
    def model_id(self) -> str:
        return 's4'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 500,
            'hidden_shape': 750,
            'dropout': 0.15,
            'learning_rate': .0001,
            'seed': 42,
            'sequence_length': 20,
            'batch_size': 10000,
        }
        return kwargs

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
            '''
        return query
