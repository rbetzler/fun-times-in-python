from data import reporter
from utilities import utils


class ShortPuts(reporter.Reporter):

    @property
    def query(self) -> str:
        query = '''
        with
        decisions as (
          select *
            , dense_rank() over (partition by model_id order by market_datetime desc) as dr
          from dbt_trade.short_puts
          where model_id = 's2'
            and has_sufficient_days_to_expiration
            and has_sufficient_pe
            and has_sufficient_market_cap
        )
        select *
        from decisions
        where dr = 1
        '''
        return query

    @property
    def email_recipients(self) -> list:
        return [
            utils.retrieve_secret('EMAIL_DAD'),
            utils.retrieve_secret('EMAIL_ME'),
        ]

    @property
    def subject(self) -> str:
        return f'Daily Short Puts Report {self.report_day}'

    @property
    def body(self) -> str:
        return "Please see the attached for today's report."

    @property
    def export_folder(self) -> str:
        return 'short_puts'

    @property
    def export_file_name(self) -> str:
        return 'short_puts'


if __name__ == '__main__':
    ShortPuts().execute()
