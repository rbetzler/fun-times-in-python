from data import reporter
from utilities import utils

REPORT_FOLDER_PREFIX = 'audit/reports'


class Options(reporter.Reporter):

    @property
    def query(self) -> str:
        query = f'''
            with
            stocks as (
              select
                  symbol
                , open
                , high
                , low
                , close
                , volume
                , market_datetime
              from dbt.stocks
              where market_datetime = (select max(market_datetime) from dbt.stocks)
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
                and open_interest > 1
                and days_to_expiration < 60
              )
            , base as (
              select
                  coalesce(s.symbol, o.symbol) as ticker
                , coalesce(s.market_datetime, o.file_datetime) as datetime
                , s.open as equity_open
                , s.high as equity_high
                , s.low as equity_low
                , s.close as equity_close
                , s.volume as equity_volume
                , o.put_call as option_type
                , o.days_to_expiration as option_days_to_expiration
                , o.strike as option_strike
                , o.bid as option_bid
                , o.ask as option_ask
                , (o.bid + o.ask) / 2 as option_bid_ask
                , o.last as option_last
                , o.open_interest as option_open_interest
              from stocks as s
              inner join options as o
                on s.symbol = o.symbol
              )
            select *
              , option_bid_ask / option_strike as option_price_versus_strike
            from base
            where option_strike < equity_high
            order by
                ticker
              , datetime
              , option_type
              , option_days_to_expiration
              , option_strike
            limit 200000
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
