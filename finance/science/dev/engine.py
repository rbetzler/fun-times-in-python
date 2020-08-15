import pandas as pd
import torch

from finance.science import engine
from finance.science.utilities import lstm_utils, science_utils

SYMBOL = 'symbol'
OPEN = 'open'
OPEN_MAX = 'open_max'
OPEN_MIN = 'open_min'
PREDICTION = 'prediction'
NORMALIZED_OPEN = 'normalized_open'
DENORMALIZED_PREDICTION = 'denormalized_prediction'


class Dev(engine.Engine):

    @property
    def model_id(self) -> str:
        return 'v0'

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
            , lagged as (
                select
                      s.symbol
                    , s.market_datetime
                    , max(s.open) over (partition by s.symbol order by s.market_datetime rows between 1 following and 31 following) as target_max_open
                    , s.open
                    , lag(s.open, 1) over (partition by s.symbol order by s.market_datetime) as open_1
                    , lag(s.open, 2) over (partition by s.symbol order by s.market_datetime) as open_2
                    , lag(s.open, 3) over (partition by s.symbol order by s.market_datetime) as open_3
                    , lag(s.open, 4) over (partition by s.symbol order by s.market_datetime) as open_4
                    , lag(s.open, 5) over (partition by s.symbol order by s.market_datetime) as open_5
                    , lag(s.open, 6) over (partition by s.symbol order by s.market_datetime) as open_6
                    , lag(s.open, 7) over (partition by s.symbol order by s.market_datetime) as open_7
                    , lag(s.open, 8) over (partition by s.symbol order by s.market_datetime) as open_8
                    , lag(s.open, 9) over (partition by s.symbol order by s.market_datetime) as open_9
                    , lag(s.open, 10) over (partition by s.symbol order by s.market_datetime) as open_10
                    , lag(s.open, 11) over (partition by s.symbol order by s.market_datetime) as open_11
                    , lag(s.open, 12) over (partition by s.symbol order by s.market_datetime) as open_12
                    , lag(s.open, 13) over (partition by s.symbol order by s.market_datetime) as open_13
                    , lag(s.open, 14) over (partition by s.symbol order by s.market_datetime) as open_14
                    , lag(s.open, 15) over (partition by s.symbol order by s.market_datetime) as open_15
                    , lag(s.open, 16) over (partition by s.symbol order by s.market_datetime) as open_16
                    , lag(s.open, 17) over (partition by s.symbol order by s.market_datetime) as open_17
                    , lag(s.open, 18) over (partition by s.symbol order by s.market_datetime) as open_18
                    , lag(s.open, 19) over (partition by s.symbol order by s.market_datetime) as open_19
                    , lag(s.open, 20) over (partition by s.symbol order by s.market_datetime) as open_20
                    , lag(s.open, 21) over (partition by s.symbol order by s.market_datetime) as open_21
                    , lag(s.open, 22) over (partition by s.symbol order by s.market_datetime) as open_22
                    , lag(s.open, 23) over (partition by s.symbol order by s.market_datetime) as open_23
                    , lag(s.open, 24) over (partition by s.symbol order by s.market_datetime) as open_24
                    , lag(s.open, 25) over (partition by s.symbol order by s.market_datetime) as open_25
                    , lag(s.open, 26) over (partition by s.symbol order by s.market_datetime) as open_26
                    , lag(s.open, 27) over (partition by s.symbol order by s.market_datetime) as open_27
                    , lag(s.open, 28) over (partition by s.symbol order by s.market_datetime) as open_28
                    , lag(s.open, 29) over (partition by s.symbol order by s.market_datetime) as open_29
                    , lag(s.open, 30) over (partition by s.symbol order by s.market_datetime) as open_30
                from td.stocks as s
                inner join tickers as t
                    on t.ticker = s.symbol
                where s.market_datetime between '{self.run_datetime.replace(month=7).date()}' and '{self.run_datetime.date()}'
                )
            , summarized as (
                select *
                    , least(open_1, open_2, open_3, open_4, open_5, open_6, open_7, open_8, open_9, open_10, open_11, open_12, open_13, open_14, open_15, open_16, open_17, open_18, open_19, open_20, open_21, open_22, open_23, open_24, open_25, open_26, open_27, open_28, open_29, open_30) as open_min
                    , greatest(open_1, open_2, open_3, open_4, open_5, open_6, open_7, open_8, open_9, open_10, open_11, open_12, open_13, open_14, open_15, open_16, open_17, open_18, open_19, open_20, open_21, open_22, open_23, open_24, open_25, open_26, open_27, open_28, open_29, open_30) as open_max
                from lagged
                where open_30 is not null
                )
            select
                      symbol
                    , market_datetime
                    , target_max_open
                    , open
                    , open_min
                    , open_max
                    , (target_max_open - open_min) / (open_max - open_min) as normalized_open
                    , (open_1 - open_min) / (open_max - open_min) as open_1
                    , (open_2 - open_min) / (open_max - open_min) as open_2
                    , (open_3 - open_min) / (open_max - open_min) as open_3
                    , (open_4 - open_min) / (open_max - open_min) as open_4
                    , (open_5 - open_min) / (open_max - open_min) as open_5
                    , (open_6 - open_min) / (open_max - open_min) as open_6
                    , (open_7 - open_min) / (open_max - open_min) as open_7
                    , (open_8 - open_min) / (open_max - open_min) as open_8
                    , (open_9 - open_min) / (open_max - open_min) as open_9
                    , (open_10 - open_min) / (open_max - open_min) as open_10
                    , (open_11 - open_min) / (open_max - open_min) as open_11
                    , (open_12 - open_min) / (open_max - open_min) as open_12
                    , (open_13 - open_min) / (open_max - open_min) as open_13
                    , (open_14 - open_min) / (open_max - open_min) as open_14
                    , (open_15 - open_min) / (open_max - open_min) as open_15
                    , (open_16 - open_min) / (open_max - open_min) as open_16
                    , (open_17 - open_min) / (open_max - open_min) as open_17
                    , (open_18 - open_min) / (open_max - open_min) as open_18
                    , (open_19 - open_min) / (open_max - open_min) as open_19
                    , (open_20 - open_min) / (open_max - open_min) as open_20
                    , (open_21 - open_min) / (open_max - open_min) as open_21
                    , (open_22 - open_min) / (open_max - open_min) as open_22
                    , (open_23 - open_min) / (open_max - open_min) as open_23
                    , (open_24 - open_min) / (open_max - open_min) as open_24
                    , (open_25 - open_min) / (open_max - open_min) as open_25
                    , (open_26 - open_min) / (open_max - open_min) as open_26
                    , (open_27 - open_min) / (open_max - open_min) as open_27
                    , (open_28 - open_min) / (open_max - open_min) as open_28
                    , (open_29 - open_min) / (open_max - open_min) as open_29
                    , (open_30 - open_min) / (open_max - open_min) as open_30
            from summarized
            where target_max_open is not null
            order by market_datetime, symbol
            '''
        return query

    @property
    def target_column(self) -> str:
        return NORMALIZED_OPEN

    @property
    def columns_to_ignore(self) -> list:
        cols = [
            'market_datetime',
            'target_max_open',
            SYMBOL,
            OPEN,
            OPEN_MIN,
            OPEN_MAX,
            NORMALIZED_OPEN,
        ]
        return cols

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = science_utils.encode_one_hot(df, [SYMBOL])
        return df

    def run_model(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        model = lstm_utils.TorchLSTM(
            x=df.drop(self.columns_to_ignore, axis=1),
            y=df[self.target_column],
            n_layers=2,
            n_training_batches=1,
            n_epochs=250,
            hidden_shape=1000,
            dropout=0.1,
            learning_rate=.0001,
            seed=44,
        )

        if self.is_training_run:
            model.fit()
            torch.save(model.state_dict(), self.trained_model_filepath)
        else:
            trained_model_params = torch.load(self.trained_model_filepath)
            model.load_state_dict(trained_model_params)

        prediction = model.prediction_df
        return prediction

    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        df = input[self.columns_to_ignore].join(output)
        df[DENORMALIZED_PREDICTION] = df[PREDICTION] * (df[OPEN_MAX] - df[OPEN_MIN]) + df[OPEN_MIN]
        return df


if __name__ == '__main__':
    Dev().execute()
