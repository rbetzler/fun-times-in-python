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
    is_call: bool
    implied_volatility: float
    theta: float
    theta_half: float
    theta_quarter: float
    theta_tenth: float
    risk_neutral_probability: float
    market_datetime: datetime.datetime


def get_report_days() -> list:
    """Get the last black scholes run date and convert the timestamp type"""
    query = '''
        with
        b as (
          select max(market_datetime)::date as bs_latest
          from td.black_scholes
        )
        , o as (
          select
              min(market_datetime)::date as option_earliest
            , max(market_datetime)::date as option_latest
          from dbt.options
        )
        select
            o.option_latest
          , coalesce(b.bs_latest, o.option_earliest) as bs_latest
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
        from dbt.holidays;
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
              from dbt.stocks
              where market_datetime = '{self._report_day}'
              )
            , options as (
              select
                  symbol
                , file_datetime
                , is_call
                , strike
                , days_to_expiration
                , last
                , ask
                , volatility
                , expiration_date_from_epoch
              from dbt.options
              where file_datetime = '{self._report_day}'
                and days_to_expiration > 0
              )
            , final as (
              select distinct
                  s.market_datetime
                , s.symbol
                , s.close
                , o.is_call
                , o.strike
                , o.days_to_expiration as days_to_maturity
                , o.last
                , o.ask
                , o.expiration_date_from_epoch
              from stocks as s
              inner join options as o
                on  s.market_datetime = o.file_datetime
                and s.symbol = o.symbol
              order by
                  s.market_datetime
                , s.symbol
                , o.is_call
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
            is_call: bool,
            market_datetime: datetime.datetime,
    ) -> tuple:
        """
        1. Calculate implied volatility
        2. Calculate theta, as well as a few forward thetas
        3. Return greek namedtuple
        """
        kwargs = {
            'current_option_price': ask,
            'stock': close,
            'strike': strike,
            'risk_free_rate': risk_free_rate,
            'days_to_maturity': days_to_maturity,
            'is_call': is_call,
        }
        bs = options_utils.BlackScholes(**kwargs)
        implied_volatility = bs.implied_volatility

        if implied_volatility:
            kwargs['volatility'] = implied_volatility
            rnp = options_utils.BlackScholes(**kwargs).risk_neutral_probability

            thetas = []
            for x in [1, 2, 4, 10]:
                kwargs['days_to_maturity'] = days_to_maturity / x
                theta = options_utils.BlackScholes(**kwargs).theta
                thetas.append(theta)

        else:
            rnp = None
            thetas = [None, None, None, None]

        greeks = Greeks(
            symbol=symbol,
            strike=strike,
            days_to_maturity=days_to_maturity,
            is_call=is_call,
            implied_volatility=implied_volatility,
            theta=thetas[0],
            theta_half=thetas[1],
            theta_quarter=thetas[2],
            theta_tenth=thetas[3],
            risk_neutral_probability=rnp,
            market_datetime=market_datetime,
        )
        return greeks

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'Starting implied vol calcs {datetime.datetime.utcnow()}')

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
                row.is_call,
                row.market_datetime,
            ): row for row in df.itertuples()
        }

        results = []
        for future in futures.as_completed(future_submission):
            results.append(future.result())

        print(f'Finished implied vol calcs {datetime.datetime.utcnow()}')
        vols = pd.DataFrame(results)

        return vols


if __name__ == '__main__':
    report_days = get_report_days()
    for report_day in report_days:
        BlackScholes(report_day=report_day).execute()
