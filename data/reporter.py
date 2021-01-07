import abc
import datetime
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from utilities import utils

REPORT_FOLDER_PREFIX = 'audit/reports'


class Reporter(abc.ABC):
    def __init__(
            self,
            report_day=datetime.datetime.utcnow(),
    ):
        self._report_day = report_day
        self.email_user = utils.retrieve_secret('EMAIL_USER')
        self.email_password = utils.retrieve_secret('EMAIL_PASSWORD')

    @property
    def report_day(self) -> str:
        return self._report_day.strftime('%Y%m%d%H%M')

    @property
    @abc.abstractmethod
    def query(self) -> str:
        pass

    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        """Helper function for non-sql data transformations"""
        return df

    @property
    def email_recipients(self) -> list:
        """To whom emails get sent"""
        pass

    @property
    def subject(self) -> str:
        """Email subject line"""
        pass

    @property
    def body(self) -> str:
        """Email body"""
        pass

    def send_emails(self, df: pd.DataFrame):
        try:

            # Connect to smtp server
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.email_user, self.email_password)

            # Construct message
            message = MIMEMultipart('alternative')
            message['Subject'] = self.subject
            message['From'] = self.email_user
            body = MIMEText(self.body, 'plain')
            message.attach(body)

            # Attach file
            attachment = MIMEText(df.to_csv())
            attachment.add_header('Content-Disposition', 'attachment', filename=f'data_{self.report_day}.csv')
            message.attach(attachment)

            # Send email
            server.sendmail(self.email_user, self.email_recipients, message.as_string())
            server.close()
            print(f'Email sent to {self.email_recipients}')

        except Exception as e:
            print(f'Email failed to send {e}')

    @property
    @abc.abstractmethod
    def export_folder(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def export_file_name(self) -> str:
        pass

    @property
    def export_file_type(self) -> str:
        return '.csv'

    @property
    def export_file_path(self) -> str:
        file_path = f'{REPORT_FOLDER_PREFIX}/{self.export_folder}/{self.export_file_name}_{self.report_day}{self.export_file_type}'
        return file_path

    def execute(self):
        print('Getting data')
        df = utils.query_db(query=self.query)

        print('Processing data')
        df = self.process_df(df)

        if self.email_recipients:
            print('Sending email')
            self.send_emails(df)

        print('Archiving report')
        df.to_csv(self.export_file_path, index=False)

        print('Report complete')
