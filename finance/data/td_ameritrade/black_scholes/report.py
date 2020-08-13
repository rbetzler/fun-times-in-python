import datetime
import pandas as pd

from concurrent import futures
from typing import NamedTuple

from finance.utilities import utils
from finance.data import reporter
from finance.science.utilities import options_utils

N_WORKERS = 15
RISK_FREE_RATE = .001


class Greeks(NamedTuple):
    symbol: str
    strike: float
    days_to_maturity: int
    put_call: bool
    implied_volatility: float


class BlackScholes(reporter.Reporter):

    @property
    def last_run_datetime(self) -> datetime.datetime:
        """Get the last black scholes run date and convert the timestamp type"""
        query = '''
            select date_trunc('day', max(file_datetime)) as last_run_date
            from td.black_scholes;
            '''
        df = utils.query_db(query=query)

        # Convert np datetime to pandas datetime to python datetime
        raw_last_run_date = df['last_run_date'].values[0]
        if raw_last_run_date:
            last_run_date = pd.Timestamp(raw_last_run_date).to_pydatetime()
        else:
            last_run_date = self._report_day
        return last_run_date

    @property
    def report_day(self) -> str:
        """Get the earlier of (a) the last black scholes run or (b) the report day specified"""
        last_run_date = self.last_run_datetime
        report_day = min(last_run_date, self._report_day).strftime('%Y%m%d')
        return report_day

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
              where market_datetime = '{self.report_day}'
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
              where file_datetime = '{self.report_day}'
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
    BlackScholes().execute()
