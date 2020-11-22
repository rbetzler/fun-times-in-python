import numpy as np
import pandas as pd
from finance.science import decisioner
from finance.science.utilities import science_utils

ASSET = 'OPTION'
DIRECTION = 'SELL'


class StockDecisioner(decisioner.Decisioner):
    @property
    def decisioner_id(self) -> str:
        return 'z2'

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
          from td.stocks
          where market_datetime >= current_date - 3
        )
        , raw_options as (
          select
              symbol
            , put_call
            , days_to_expiration
            , strike
            , (bid + ask)/2 as price
            , row_number() over (w) as rn
          from td.options
          where file_datetime >= current_date - 3
            and days_to_expiration between 10 and 60
            and put_call = 'PUT'
          window w as (partition by symbol, strike, days_to_expiration, put_call order by file_datetime desc)
        )
        , options as (
            select *
                , (lead(price) over (w) - price) / (lead(strike) over (w) - strike) as first_order_difference
            from raw_options
            window w as (partition by symbol, days_to_expiration order by strike)
        )
        , final as (
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
            and s.rn = 1
            and o.rn = 1
          where p.dr = 1
        )
        select *
          , potential_annual_return > .2 as is_sufficiently_profitable
          , oom_percent > .15 as is_sufficiently_oom
          , strike < thirty_day_low_prediction as is_strike_below_predicted_low_price
        from final
        order by symbol, days_to_expiration, strike
        '''
        return query

    @staticmethod
    def smooth_first_order_difference(
            df: pd.DataFrame,
            degree: int = 2,
    ) -> pd.DataFrame:
        """
        Smooth the first order differences of an option chain. Recall that an
        option's first order difference is its cumulative density function.
        """
        p = np.polyfit(
            x=df['strike'],
            y=df['first_order_difference'],
            deg=degree,
        )
        df['smoothed_first_order_difference'] = np.polyval(p, df['strike'])
        df.loc[df['smoothed_first_order_difference'] < 0, 'smoothed_first_order_difference'] = 0
        df.loc[df['smoothed_first_order_difference'] > 1, 'smoothed_first_order_difference'] = 1
        return df

    @staticmethod
    def select_trades(
        df: pd.DataFrame,
        n_stocks: int = 10,
        n_trades: int = 5,
    ) -> pd.DataFrame:
        """
        Select trades to place: Of the ten stocks with the highest average
        kelly, pick the five most profitable trades.
        """
        x = df[
            df['is_sufficiently_profitable'] &
            df['is_sufficiently_oom'] &
            df['is_strike_below_predicted_low_price'] &
            df['kelly_criterion'] > 0
        ]
        symbols = x.groupby('symbol')['kelly_criterion'].mean().nlargest(n_stocks).index
        x_symbols = x[x['symbol'].isin(symbols)]
        idx = x_symbols.groupby('symbol')['kelly_criterion'].transform(max) == x_symbols['kelly_criterion']
        trades = x_symbols[idx].nlargest(n_trades, 'kelly_criterion')
        df.loc[df.index.isin(trades.index), 'quantity'] = 1
        return df

    def decision(self, df: pd.DataFrame) -> pd.DataFrame:
        df['quantity'] = 0
        df['asset'] = ASSET
        df['direction'] = DIRECTION

        print('Smoothing first order differences')
        df = df.groupby(['symbol', 'days_to_expiration']).apply(self.smooth_first_order_difference)

        print('Calculating kelly criterion')
        df['kelly_criterion'] = science_utils.kelly_criterion(
            predicted_win=df['price'],
            predicted_loss=df['strike'],
            p_win=df['smoothed_first_order_difference'],
        )

        print('Finalize trades to place')
        trades = self.select_trades(df)
        return trades


if __name__ == '__main__':
    StockDecisioner(model_id='s0').execute()
