import abc
import pandas as pd

from science.models import lstm, nn
from science.predictor import thirty_day_low as tdl
from utilities import science_utils, utils

PREDICTION = 'prediction'
SCALED_PREDICTION = 'scaled_prediction'
TARGET = 'target'
SCALED_TARGET = 'scaled_target'

MARKET_DATETIME = 'market_datetime'
SYMBOL = 'symbol'
SECTOR = 'sector'
INDUSTRY = 'industry'
CLOSE = 'close'
SCALED_TARGET_MIN = 'scaled_target_min'
SCALED_TARGET_MAX = 'scaled_target_max'
TARGET_MIN = 'target_min'
TARGET_MAX = 'target_max'


class SpeculativePredictor(tdl.ThirtyDayLowPredictor, abc.ABC):
    """Subclass for speculative predictions"""

    @property
    def columns_to_ignore(self) -> list:
        cols = [
            MARKET_DATETIME,
            SYMBOL,
            SECTOR,
            INDUSTRY,
            CLOSE,
            SCALED_TARGET_MIN,
            SCALED_TARGET_MAX,
            TARGET_MIN,
            TARGET_MAX,
        ]
        return cols

    @property
    def query(self) -> str:
        query = f'''
            select
                market_datetime
              , symbol
              , sector
              , industry
              , close
              , scaled_target_min
              , scaled_target_max
              , target_min
              , target_max
              , daily_range_1
              , daily_range_2
              , daily_range_3
              , daily_range_5
              , daily_range_10
              , daily_range_20
              , daily_range_30
              , intraday_performance_1
              , intraday_performance_2
              , intraday_performance_3
              , intraday_performance_5
              , intraday_performance_10
              , intraday_performance_20
              , intraday_performance_30
              , open_over_prior_close_1
              , open_over_prior_close_2
              , open_over_prior_close_3
              , open_over_prior_close_5
              , open_over_prior_close_10
              , open_over_prior_close_20
              , open_over_prior_close_30
              , open_over_10_day_max_1
              , open_over_10_day_max_2
              , open_over_10_day_max_3
              , open_over_10_day_max_5
              , open_over_10_day_max_10
              , open_over_10_day_max_20
              , open_over_10_day_max_30
              , open_over_30_day_max_1
              , open_over_30_day_max_2
              , open_over_30_day_max_3
              , open_over_30_day_max_5
              , open_over_30_day_max_10
              , open_over_30_day_max_20
              , open_over_30_day_max_30
              , open_over_60_day_max_1
              , open_over_60_day_max_2
              , open_over_60_day_max_3
              , open_over_60_day_max_5
              , open_over_60_day_max_10
              , open_over_60_day_max_20
              , open_over_60_day_max_30
              , open_over_90_day_max_1
              , open_over_90_day_max_2
              , open_over_90_day_max_3
              , open_over_90_day_max_5
              , open_over_90_day_max_10
              , open_over_90_day_max_20
              , open_over_90_day_max_30
              , volume_over_10_day_max_1
              , volume_over_10_day_max_2
              , volume_over_10_day_max_3
              , volume_over_10_day_max_5
              , volume_over_10_day_max_10
              , volume_over_10_day_max_20
              , volume_over_10_day_max_30
              , volume_over_30_day_max_1
              , volume_over_30_day_max_2
              , volume_over_30_day_max_3
              , volume_over_30_day_max_5
              , volume_over_30_day_max_10
              , volume_over_30_day_max_20
              , volume_over_30_day_max_30
              , volume_over_60_day_max_1
              , volume_over_60_day_max_2
              , volume_over_60_day_max_3
              , volume_over_60_day_max_5
              , volume_over_60_day_max_10
              , volume_over_60_day_max_20
              , volume_over_60_day_max_30
              , volume_over_90_day_max_1
              , volume_over_90_day_max_2
              , volume_over_90_day_max_3
              , volume_over_90_day_max_5
              , volume_over_90_day_max_10
              , volume_over_90_day_max_20
              , volume_over_90_day_max_30
            from dbt_train.speculation
            where market_datetime between '{self.start_date}' and '{self.end_date}'
              {'and target_min is not null' if self.is_training_run else ''}
              {'and target_max is not null' if self.is_training_run else ''}
            order by market_datetime, symbol
            { 'limit ' + str(self.limit) if self.is_training_run else '' }
            { 'offset ' + str(self.limit * self.n_subrun) if self.is_training_run else '' }
            '''
        return query

    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        output['model_id'] = self.model_id
        df = input[self.columns_to_ignore].join(output)
        df[SCALED_PREDICTION] = df[PREDICTION] * df[CLOSE]
        df[TARGET] = df[self.target_column]
        df[SCALED_TARGET] = df[f'scaled_{self.target_column}']
        return df


class LowSpeculativePredictorNN(SpeculativePredictor, abc.ABC):
    """NN implementation of 10 day low price speculative predictor"""

    @property
    def target_column(self) -> str:
        return TARGET_MIN

    def model(self, df: pd.DataFrame):
        model = lstm.LSTM0(
            x=df.drop(self.columns_to_ignore, axis=1),
            y=df[self.target_column],
            device=self.device,
            **self.model_kwargs,
        )
        return model


class HighSpeculativePredictorNN(SpeculativePredictor, abc.ABC):
    """NN implementation of 10 day high price speculative predictor"""

    @property
    def target_column(self) -> str:
        return TARGET_MAX

    def model(self, df: pd.DataFrame):
        model = nn.NN0(
            x=df.drop(self.columns_to_ignore, axis=1),
            y=df[self.target_column],
            device=self.device,
            **self.model_kwargs,
        )
        return model
