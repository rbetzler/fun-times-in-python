from finance.data import reporter
from finance.science.dev import decisioner
from finance.utilities import utils

REPORT_FOLDER_PREFIX = 'audit/reports'


class Decisions(reporter.Reporter):

    @property
    def query(self) -> str:
        return decisioner.StockDecisioner(model_id='s0').query

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
