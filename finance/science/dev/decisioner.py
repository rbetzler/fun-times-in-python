import pandas as pd
from finance.science import decisioner
from finance.science.utilities import science_utils

ASSET = 'OPTION'
DIRECTION = 'SELL'


class StockDecisioner(decisioner.Decisioner):
    @property
    def decisioner_id(self) -> str:
        return 'z0'

    @property
    def query(self) -> str:
        query = f'''
        with
        predictions as (
          select *
            , dense_rank() over (partition by model_id order by file_datetime desc) as dr
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
        , options as (
          select
              symbol
            , put_call
            , days_to_expiration
            , strike
            , (bid + ask)/2 as price
            , row_number() over (partition by symbol, strike, days_to_expiration, put_call order by file_datetime desc) as rn
          from td.options
          where file_datetime >= current_date - 3
          and days_to_expiration between 10 and 60
          and put_call = 'PUT'
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

    def decision(self, df: pd.DataFrame) -> pd.DataFrame:
        df['quantity'] = 0
        df['asset'] = ASSET
        df['direction'] = DIRECTION
        df['kelly_criterion'] = science_utils.kelly_criterion(
            predicted_win=df['price'],
            predicted_loss=df['strike'],
            p_win=df['oom_percent'],
        )
        potential_trades = df[
            df['is_sufficiently_profitable'] & df['is_sufficiently_oom'] & df['is_strike_below_predicted_low_price']]
        potential_trades = potential_trades.reset_index()
        return potential_trades


if __name__ == '__main__':
    StockDecisioner(model_id='s0').execute()
