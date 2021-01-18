import abc
import numpy as np
import pandas as pd
from science.decisioner import base
from science.utilities import options_utils, science_utils

ASSET = 'OPTION'
DIRECTION = 'SELL'


class D1(base.Decisioner, abc.ABC):
    @property
    def decisioner_id(self) -> str:
        return 'd1'

    @property
    def model_id(self) -> str:
        return 's1'

    @property
    def query(self) -> str:
        query = f'''
        with
        predictions as (
          select *
            , dense_rank() over (partition by model_id order by file_datetime desc, market_datetime desc) as dr
          from dev.predictions
          where model_id = '{self.model_id}'
        )
        , stocks as (
          select
              market_datetime
            , symbol
            , close
            , row_number() over (partition by symbol order by market_datetime desc) as rn
          from dbt.stocks
          where market_datetime between '{self.start_date}'::date - 5 and '{self.start_date}'
        )
        , raw_options as (
          select
              symbol
            , put_call
            , days_to_expiration
            , strike
            , (bid + ask)/2 as price
            , dense_rank() over (w) as dr
          from dbt.options
          where file_datetime between '{self.start_date}'::date - 5 and '{self.start_date}'
            and days_to_expiration between 10 and 60
            and put_call = 'PUT'
          window w as (partition by symbol order by file_datetime desc)
        )
        , options as (
            select *
                , (lead(price) over (w) - price) / (lead(strike) over (w) - strike) as first_order_difference
            from raw_options
            where dr = 1
            window w as (partition by symbol, days_to_expiration order by strike)
        )
        , base as (
          select
              p.model_id
            , p.file_datetime as model_datetime
            , p.market_datetime
            , p.symbol
            , p.denormalized_prediction as thirty_day_low_prediction
            , s.close
            , o.put_call
            , o.days_to_expiration
            , o.strike
            , o.price
            , greatest(.01, least(.99, o.first_order_difference)) as first_order_difference
            , (o.price / o.strike) * (360 / o.days_to_expiration) as potential_annual_return
            , (s.close - o.strike)/s.close as oom_percent
          from predictions as p
          inner join stocks as s
            on  p.symbol = s.symbol
          inner join options as o
            on  s.symbol = o.symbol
          where p.dr = 1
            and s.rn = 1
            and o.dr = 1
        )
        , final as (
            select *
              , potential_annual_return > .2 as is_sufficiently_profitable
              , oom_percent > .15 as is_sufficiently_oom
              , strike < thirty_day_low_prediction as is_strike_below_predicted_low_price
            from base
            order by symbol, days_to_expiration, strike
        )
        select *
        from final
        '''
        return query

    @staticmethod
    def select_trades(
        df: pd.DataFrame,
        n_stocks: int = 250,
        n_trades: int = 250,
    ) -> pd.DataFrame:
        """
        Select trades to place: Of the ten stocks with the highest average
        kelly, pick the five most profitable trades.
        """

        temp = df[
            df['is_sufficiently_profitable'] &
            df['is_sufficiently_oom'] &
            df['is_strike_below_predicted_low_price'] &
            df['kelly_criterion'] > 0
        ]

        symbols = temp.groupby('symbol')['kelly_criterion'].mean().nlargest(n_stocks).index
        x = temp[temp['symbol'].isin(symbols)]
        idx = x.groupby('symbol')['kelly_criterion'].transform(max) == x['kelly_criterion']
        df.loc[df.index.isin(idx[idx].index), 'quantity'] = 1
        return df

    def decision(self, df: pd.DataFrame) -> pd.DataFrame:
        df['quantity'] = 0
        df['asset'] = ASSET
        df['direction'] = DIRECTION

        print('Smoothing first order differences')
        df = df.groupby(['symbol', 'days_to_expiration']).apply(options_utils.smooth_first_order_difference)

        print('Calculating kelly criterion')
        df['kelly_criterion'] = science_utils.kelly_criterion(
            predicted_win=df['price'],
            predicted_loss=df['strike'],
            p_win=df['probability_of_profit'],
        )
        df.loc[df['kelly_criterion'] == -np.inf, 'kelly_criterion'] = None

        print('Finalize trades to place')
        trades = self.select_trades(df)
        return trades
