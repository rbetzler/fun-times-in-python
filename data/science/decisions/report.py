from data import reporter
from science.decisioner import d1
from utilities import utils

REPORT_FOLDER_PREFIX = 'audit/reports'


class Decisions(reporter.Reporter):

    @property
    def query(self) -> str:
        query = '''
        with
        decisions as (
            select *
                , dense_rank() over (order by file_datetime desc, market_datetime desc) as dr
            from dev.decisions
            where model_id = 's1'
              and decisioner_id = 'd1'
        )
        select
              d.model_id
            , d.decisioner_id
            , d.model_datetime
            , d.market_datetime
            , d.symbol
            , d.thirty_day_low_prediction
            , d.close as closing_stock_price
            , d.put_call as option_type
            , d.days_to_expiration
            , d.strike
            , d.price as option_price
            , d.potential_annual_return
            , d.oom_percent
            , d.quantity > 0 as should_place_trade
            , d.direction
            , 1 - d.first_order_difference as raw_probability_of_profit
            , 1 - d.smoothed_first_order_difference as adj_probability_of_profit
            , d.kelly_criterion
            , f.pe_ratio
            , f.dividend_amount
            , f.dividend_date
            , f.market_capitalization
            , f.eps_ttm as eps_trailing_twelve_months
            , f.quick_ratio
            , f.current_ratio
            , f.total_debt_to_equity
            , t.avg_open_10
            , t.avg_open_20
            , t.avg_open_30
            , t.avg_open_60
            , t.avg_open_90
            , t.avg_open_120
            , t.avg_open_180
            , t.avg_open_240
            , b.implied_volatility
        from decisions as d
        left join dbt.fundamentals as f
          on  d.symbol = f.symbol
          and d.market_datetime = f.market_datetime
        left join dbt.technicals as t
          on  d.symbol = t.symbol
          and d.market_datetime = t.market_datetime
        left join dbt.black_scholes as b
          on  d.symbol = b.symbol
          and d.market_datetime = b.market_datetime
          and d.strike = b.strike
          and d.days_to_expiration = b.days_to_maturity
          and not b.is_call
        where d.dr = 1
        order by d.symbol, d.days_to_expiration, d.strike
        '''
        return query

    @property
    def email_recipients(self) -> list:
        return utils.retrieve_secret('EMAIL_DAD')

    @property
    def subject(self) -> str:
        return f'Daily Decision Report {self.report_day}'

    @property
    def body(self) -> str:
        return "Please see the attached for today's report."

    @property
    def export_folder(self) -> str:
        return 'decisions'

    @property
    def export_file_name(self) -> str:
        return 'decisions'


if __name__ == '__main__':
    Decisions().execute()
