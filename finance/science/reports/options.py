from finance.science.reports import reporter
from finance.utilities import utils

REPORT_FOLDER_PREFIX = 'audit/reports'


class Options(reporter.Reporter):

    @property
    def query(self) -> str:
        query = f'''
            with
            equities as (
              select
                  symbol
                , open
                , high
                , low
                , close
                , volume
                , market_datetime::date as market_datetime
              from td.equities
              where market_datetime = (select max(market_datetime) from td.equities)
              )
            , options as (
              select
                  symbol
                , put_call
                , strike
                , days_to_expiration
                , bid
                , ask
                , last
                , open_interest
                , file_datetime
              from td.options
              where file_datetime >= (select max(file_datetime)::date from td.options)
                and put_call = 'PUT'
                and open_interest > 0
                and days_to_expiration < 150
              )
            , base as (
              select
                  coalesce(e.symbol, o.symbol) as ticker
                , coalesce(e.market_datetime, o.file_datetime) as datetime
                , e.open as equity_open
                , e.high as equity_high
                , e.low as equity_low
                , e.close as equity_close
                , e.volume as equity_volume
                , o.put_call as option_type
                , o.days_to_expiration as option_days_to_expiration
                , o.strike as option_strike
                , o.bid as option_bid
                , o.ask as option_ask
                , (o.bid + o.ask) / 2 as option_bid_ask
                , o.last as option_last
                , o.open_interest as option_open_interest
              from equities as e
              inner join options as o
                on e.symbol = o.symbol
              )
            select *
              , option_bid_ask / option_strike as option_price_versus_strike
            from base
            order by
                ticker
              , datetime
              , option_type
              , option_days_to_expiration
              , option_strike
            '''
        return query

    @property
    def email_recipients(self) -> list:
        return utils.retrieve_secret('EMAIL_DAD')

    @property
    def subject(self) -> str:
        return f'Daily Options Report {self.report_day}'

    @property
    def body(self) -> str:
        return "Please see the attached for today's report."

    @property
    def export_folder(self) -> str:
        return 'options'

    @property
    def export_file_name(self) -> str:
        return 'options'


if __name__ == '__main__':
    Options().execute()
