import datetime
import pandas as pd

from concurrent import futures
from typing import NamedTuple

from utilities import utils
from data import reporter
from science.utilities import options_utils

N_WORKERS = 15
RISK_FREE_RATE = .001


class Greeks(NamedTuple):
    symbol: str
    strike: float
    days_to_maturity: int
    put_call: bool
    implied_volatility: float
    market_datetime: datetime.datetime


def get_report_days() -> list:
    """Get the last black scholes run date and convert the timestamp type"""
    query = '''
        with
        b as (
          select max(file_datetime)::date as bs_latest
          from td.black_scholes
        )
        , o as (
          select max(file_datetime)::date as option_latest
          from td.options_raw
          where file_datetime > current_date - 5
        )
        select
            o.option_latest
          , b.bs_latest + 1 as bs_latest
        from b, o;
        '''
    df = utils.query_db(query=query)

    days = pd.date_range(
        start=df['bs_latest'].values[0],
        end=df['option_latest'].values[0],
        freq='b'
    ).to_frame(index=False, name='dates')

    query = '''
        select distinct day_date
        from utils.holidays;
        '''
    holidays = utils.query_db(query=query)

    report_days = days.loc[~days['dates'].isin(holidays['day_date']), 'dates'].to_list()
    return report_days


class BlackScholes(reporter.Reporter):

    @property
    def export_folder(self) -> str:
        return 'black_scholes'

    @property
    def export_file_name(self) -> str:
        return 'black_scholes'

    @property
    def query(self) -> str:
        query = f'''
            with
            stocks as (
              select
                  market_datetime
                , symbol
                , close
              from td.stocks
              where market_datetime = '{self._report_day}'
              )
            , options as (
              select
                  symbol
                , file_datetime
                , put_call
                , strike
                , days_to_expiration
                , last
                , ask
                , volatility
                , expiration_date_from_epoch
              from td.options
              where file_datetime = '{self._report_day}'
                and days_to_expiration > 0
              )
            , final as (
              select distinct
                  s.market_datetime
                , s.symbol
                , s.close
                , o.put_call
                , o.strike
                , o.days_to_expiration as days_to_maturity
                , o.last
                , o.ask
                , o.volatility
                , o.expiration_date_from_epoch
              from stocks as s
              inner join options as o
                on  s.market_datetime = o.file_datetime
                and s.symbol = o.symbol
              order by
                  s.market_datetime
                , s.symbol
                , o.put_call
                , o.strike
                , o.days_to_expiration
              )
            select *
            from final;
            '''
        return query

    @staticmethod
    def executor_helper(
            symbol: str,
            ask: float,
            close: float,
            strike: float,
            risk_free_rate: float,
            days_to_maturity: int,
            put_call: bool,
            market_datetime: datetime.datetime,
    ) -> tuple:
        bs = options_utils.BlackScholes(
            current_option_price=ask,
            stock_price=close,
            strike=strike,
            risk_free_rate=risk_free_rate,
            days_to_maturity=days_to_maturity,
            is_call=put_call,
        )
        greek = Greeks(
            symbol=symbol,
            strike=strike,
            days_to_maturity=days_to_maturity,
            put_call=put_call,
            implied_volatility=bs.implied_volatility,
            market_datetime=market_datetime,
        )
        return greek

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'Starting implied vol calcs {datetime.datetime.utcnow()}')

        # TODO: Pass thru market_datetime to csv
        executor = futures.ProcessPoolExecutor(max_workers=N_WORKERS)
        future_submission = {
            executor.submit(
                self.executor_helper,
                row.symbol,
                row.ask,
                row.close,
                row.strike,
                RISK_FREE_RATE,
                row.days_to_maturity,
                row.put_call == 'CALL',
                row.market_datetime,
            ): row for row in df.itertuples()
        }

        results = []
        for future in futures.as_completed(future_submission):
            results.append(future.result())

        print(f'Finished implied vol calcs {datetime.datetime.utcnow()}')
        vols = pd.DataFrame(results)
        vols = vols.dropna()

        return vols


if __name__ == '__main__':
    report_days = get_report_days()
    for report_day in report_days:
        BlackScholes(report_day=report_day).execute()
