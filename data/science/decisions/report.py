from data import reporter
from science.dev import decisioner
from utilities import utils

REPORT_FOLDER_PREFIX = 'audit/reports'


class Decisions(reporter.Reporter):

    @property
    def query(self) -> str:
        query = '''
        with
        raw as (
            select *
                , dense_rank() over (order by file_datetime desc, market_datetime desc) as dr
            from dev.decisions
            where model_id = 's1'
              and decisioner_id = 'z2'
        )
        select
              model_id
            , decisioner_id
            , model_datetime
            , market_datetime
            , symbol
            , thirty_day_low_prediction
            , close as closing_stock_price
            , put_call as option_type
            , days_to_expiration
            , strike
            , price as option_price
            , potential_annual_return
            , oom_percent
            , is_sufficiently_profitable
            , is_sufficiently_oom
            , is_strike_below_predicted_low_price
            , quantity > 0 as should_place_trade
            , direction
            , 1 - first_order_difference as raw_probability_of_profit
            , 1 - smoothed_first_order_difference as adj_probability_of_profit
            , kelly_criterion
        from raw
        where dr = 1
        order by symbol, days_to_expiration, strike
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
