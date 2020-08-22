import datetime
import pandas as pd

from finance.science import predictor
from finance.science.utilities import science_utils

SYMBOL = 'symbol'
TARGET = 'target'


class VolatilityPredictor(predictor.Predictor):
    """Predict volatility over the next 30 days"""

    @property
    def model_id(self) -> str:
        return 'v0'

    # TODO: Check if open_0 date offset causes information leakage
    @property
    def query(self) -> str:
        query = f'''
            with
            tickers as (
                select distinct
                      ticker
                    , sector
                    , industry
                from nasdaq.listed_stocks
                where   ticker !~ '[\^.~]'
                    and character_length(ticker) between 1 and 4
                    and ticker in ('KO', 'JPM', 'AA')
                limit 30
                )
            , lagged_prices as (
                select
                      s.symbol
                    , s.market_datetime
                    , avg(s.open) over (partition by s.symbol order by s.market_datetime rows between 31 preceding and 1 preceding) as mean
                    , s.open as open_0
                    , lag(s.open,  1) over (w) as open_1
                    , lag(s.open,  2) over (w) as open_2
                    , lag(s.open,  3) over (w) as open_3
                    , lag(s.open,  4) over (w) as open_4
                    , lag(s.open,  5) over (w) as open_5
                    , lag(s.open,  6) over (w) as open_6
                    , lag(s.open,  7) over (w) as open_7
                    , lag(s.open,  8) over (w) as open_8
                    , lag(s.open,  9) over (w) as open_9
                    , lag(s.open, 10) over (w) as open_10
                    , lag(s.open, 11) over (w) as open_11
                    , lag(s.open, 12) over (w) as open_12
                    , lag(s.open, 13) over (w) as open_13
                    , lag(s.open, 14) over (w) as open_14
                    , lag(s.open, 15) over (w) as open_15
                    , lag(s.open, 16) over (w) as open_16
                    , lag(s.open, 17) over (w) as open_17
                    , lag(s.open, 18) over (w) as open_18
                    , lag(s.open, 19) over (w) as open_19
                    , lag(s.open, 20) over (w) as open_20
                    , lag(s.open, 21) over (w) as open_21
                    , lag(s.open, 22) over (w) as open_22
                    , lag(s.open, 23) over (w) as open_23
                    , lag(s.open, 24) over (w) as open_24
                    , lag(s.open, 25) over (w) as open_25
                    , lag(s.open, 26) over (w) as open_26
                    , lag(s.open, 27) over (w) as open_27
                    , lag(s.open, 28) over (w) as open_28
                    , lag(s.open, 29) over (w) as open_29
                    , lag(s.open, 30) over (w) as open_30
                from td.stocks as s
                inner join tickers as t
                    on t.ticker = s.symbol
                where s.market_datetime between '{self.start_date}' and '{self.end_date}'
                window w as (partition by s.symbol order by s.market_datetime)
                )
            , raw_deviations as (
                select
                    symbol
                  , market_datetime
                  , abs(open_0  - mean) as abs_dev_0
                  , abs(open_1  - mean) as abs_dev_1
                  , abs(open_2  - mean) as abs_dev_2
                  , abs(open_3  - mean) as abs_dev_3
                  , abs(open_4  - mean) as abs_dev_4
                  , abs(open_5  - mean) as abs_dev_5
                  , abs(open_6  - mean) as abs_dev_6
                  , abs(open_7  - mean) as abs_dev_7
                  , abs(open_8  - mean) as abs_dev_8
                  , abs(open_9  - mean) as abs_dev_9
                  , abs(open_10 - mean) as abs_dev_10
                  , abs(open_11 - mean) as abs_dev_11
                  , abs(open_12 - mean) as abs_dev_12
                  , abs(open_13 - mean) as abs_dev_13
                  , abs(open_14 - mean) as abs_dev_14
                  , abs(open_15 - mean) as abs_dev_15
                  , abs(open_16 - mean) as abs_dev_16
                  , abs(open_17 - mean) as abs_dev_17
                  , abs(open_18 - mean) as abs_dev_18
                  , abs(open_19 - mean) as abs_dev_19
                  , abs(open_20 - mean) as abs_dev_20
                  , abs(open_21 - mean) as abs_dev_21
                  , abs(open_22 - mean) as abs_dev_22
                  , abs(open_23 - mean) as abs_dev_23
                  , abs(open_24 - mean) as abs_dev_24
                  , abs(open_25 - mean) as abs_dev_25
                  , abs(open_26 - mean) as abs_dev_26
                  , abs(open_27 - mean) as abs_dev_27
                  , abs(open_28 - mean) as abs_dev_28
                  , abs(open_29 - mean) as abs_dev_29
                  , abs(open_30 - mean) as abs_dev_30
                from lagged_prices
                )
            , deviations as (
              select
                  symbol
                , market_datetime
                , (  abs_dev_0
                   + abs_dev_1
                   + abs_dev_2
                   + abs_dev_3
                   + abs_dev_4
                   + abs_dev_5
                   + abs_dev_6
                   + abs_dev_7
                   + abs_dev_8
                   + abs_dev_9
                   + abs_dev_10
                   + abs_dev_11
                   + abs_dev_12
                   + abs_dev_13
                   + abs_dev_14
                   + abs_dev_15
                   + abs_dev_16
                   + abs_dev_17
                   + abs_dev_18
                   + abs_dev_19
                   + abs_dev_20
                   + abs_dev_21
                   + abs_dev_22
                   + abs_dev_23
                   + abs_dev_24
                   + abs_dev_25
                   + abs_dev_26
                   + abs_dev_27
                   + abs_dev_28
                   + abs_dev_29
                   + abs_dev_30
                 ) / 31 as mean_abs_dev
              from raw_deviations
              )
            , final as (
                select
                    symbol
                  , market_datetime
                  , mean_abs_dev as target
                  , lag(mean_abs_dev,  1) over (w) as mean_abs_dev_1
                  , lag(mean_abs_dev,  2) over (w) as mean_abs_dev_2
                  , lag(mean_abs_dev,  3) over (w) as mean_abs_dev_3
                  , lag(mean_abs_dev,  4) over (w) as mean_abs_dev_4
                  , lag(mean_abs_dev,  5) over (w) as mean_abs_dev_5
                  , lag(mean_abs_dev,  6) over (w) as mean_abs_dev_6
                  , lag(mean_abs_dev,  7) over (w) as mean_abs_dev_7
                  , lag(mean_abs_dev,  8) over (w) as mean_abs_dev_8
                  , lag(mean_abs_dev,  9) over (w) as mean_abs_dev_9
                  , lag(mean_abs_dev, 10) over (w) as mean_abs_dev_10
                  , lag(mean_abs_dev, 11) over (w) as mean_abs_dev_11
                  , lag(mean_abs_dev, 12) over (w) as mean_abs_dev_12
                  , lag(mean_abs_dev, 13) over (w) as mean_abs_dev_13
                  , lag(mean_abs_dev, 14) over (w) as mean_abs_dev_14
                  , lag(mean_abs_dev, 15) over (w) as mean_abs_dev_15
                  , lag(mean_abs_dev, 16) over (w) as mean_abs_dev_16
                  , lag(mean_abs_dev, 17) over (w) as mean_abs_dev_17
                  , lag(mean_abs_dev, 18) over (w) as mean_abs_dev_18
                  , lag(mean_abs_dev, 19) over (w) as mean_abs_dev_19
                  , lag(mean_abs_dev, 20) over (w) as mean_abs_dev_20
                  , lag(mean_abs_dev, 21) over (w) as mean_abs_dev_21
                  , lag(mean_abs_dev, 22) over (w) as mean_abs_dev_22
                  , lag(mean_abs_dev, 23) over (w) as mean_abs_dev_23
                  , lag(mean_abs_dev, 24) over (w) as mean_abs_dev_24
                  , lag(mean_abs_dev, 25) over (w) as mean_abs_dev_25
                  , lag(mean_abs_dev, 26) over (w) as mean_abs_dev_26
                  , lag(mean_abs_dev, 27) over (w) as mean_abs_dev_27
                  , lag(mean_abs_dev, 28) over (w) as mean_abs_dev_28
                  , lag(mean_abs_dev, 29) over (w) as mean_abs_dev_29
                  , lag(mean_abs_dev, 30) over (w) as mean_abs_dev_30
                from deviations
                window w as (partition by symbol order by market_datetime)
                )
            select *
            from final
            where mean_abs_dev_30 is not null
            order by 1,2
            '''
        return query

    @property
    def target_column(self) -> str:
        return TARGET

    @property
    def columns_to_ignore(self) -> list:
        cols = [
            'market_datetime',
            SYMBOL,
        ] + [self.target_column]
        return cols

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = science_utils.encode_one_hot(df, [SYMBOL])
        return df

    @property
    def model_args(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_training_batches': 1,
            'n_epochs': 250,
            'hidden_shape': 1000,
            'dropout': 0.1,
            'learning_rate': .0001,
            'seed': 44,
        }
        return kwargs

    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        output['model_id'] = self.model_id
        df = input[self.columns_to_ignore].join(output)
        return df


if __name__ == '__main__':
    VolatilityPredictor().execute()
