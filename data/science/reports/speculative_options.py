from data import reporter
from utilities import utils


class SpeculativeOptions(reporter.Reporter):

    @property
    def query(self) -> str:
        query = '''
        select *
        from dbt.speculative_options
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
        return f'Daily Speculative Options Report {self.report_day}'

    @property
    def body(self) -> str:
        return "Please see the attached for today's report."

    @property
    def export_folder(self) -> str:
        return 'speculative_options'

    @property
    def export_file_name(self) -> str:
        return 'speculative_options'


if __name__ == '__main__':
    SpeculativeOptions().execute()
