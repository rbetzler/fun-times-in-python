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
            , dense_rank() over (partition by model_id order by market_datetime desc) as dr
          from dbt.decisions
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
