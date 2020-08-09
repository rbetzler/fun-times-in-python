import pandas as pd

from concurrent import futures
from datetime import datetime

from finance.data import reporter
from finance.science.utilities import options_utils

N_WORKERS = 15
RISK_FREE_RATE = .001


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
              where market_datetime > (select max(file_datetime)::date from td.black_scholes)
              )
            , options as (
              select
                  symbol
                , file_datetime
                , put_call
                , strike
                , days_to_expiration
                , last
                , (bid + ask) / 2 as bid_ask
                , volatility
                , expiration_date_from_epoch
              from td.options
              where file_datetime > (select max(file_datetime)::date from td.black_scholes)
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
                , o.bid_ask
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
            from final
            '''
        return query

    @staticmethod
    def executor_helper(
            symbol: str,
            bid_ask: float,
            close: float,
            strike: float,
            risk_free_rate: float,
            days_to_maturity: float,
            put_call: str,
    ) -> tuple:
        volatility = options_utils.BlackScholes(
            current_option_price=bid_ask,
            stock_price=close,
            strike=strike,
            risk_free_rate=risk_free_rate,
            days_to_maturity=days_to_maturity,
            is_call=put_call == 'CALL',
        ).implied_volatility
        return symbol, volatility, strike, days_to_maturity, put_call

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'Starting implied vol calcs {datetime.utcnow()}')

        executor = futures.ProcessPoolExecutor(max_workers=N_WORKERS)
        future_submission = {
            executor.submit(
                self.executor_helper,
                row.symbol,
                row.bid_ask,
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

        print(f'Finished implied vol calcs {datetime.utcnow()}')
        vols = pd.DataFrame(results, columns=['symbol', 'implied_volatility', 'strike', 'days_to_maturity', 'put_call'])
        vols = vols.dropna()

        return vols


if __name__ == '__main__':
    BlackScholes().execute()
