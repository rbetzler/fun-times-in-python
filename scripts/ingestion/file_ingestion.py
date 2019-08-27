import os
import abc
import datetime
import pandas as pd
from sqlalchemy import create_engine
from scripts.utilities import db_utilities


class FileIngestion(abc.ABC):
    def __init__(self,
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d')):
        self.db_connection = db_utilities.DW_STOCKS
        self.start_date = start_date
        self.end_date = end_date

    @property
    def run_date(self) -> str:
        return datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')

    @property
    def place_raw_file(self) -> bool:
        return False

    @property
    def place_batch_file(self) -> bool:
        return False

    @property
    def job_name(self) -> str:
        return 'job'

    @property
    def place_with_index(self) -> bool:
        return False

    @property
    def import_directory(self) -> str:
        return ''

    @property
    def import_file_prefix(self) -> str:
        return ''

    @property
    def import_file_extension(self) -> str:
        return '.csv'

    @property
    def import_file_date_format(self) -> str:
        return '%Y%m%d%H%M%S'

    @property
    def export_folder(self) -> str:
        return ''

    @property
    def export_file_name(self) -> str:
        return ''

    @property
    def export_file_type(self) -> str:
        return '.csv'

    def export_file_path(self, api) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + api + '_' \
                    + self.run_date \
                    + self.export_file_type
        return file_path

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def schema(self) -> str:
        return ''

    @property
    def db_engine(self) -> str:
        return create_engine(self.db_connection)

    @property
    def append_to_table(self) -> str:
        return 'append'

    @property
    def column_mapping(self) -> dict:
        return {}

    @property
    def index(self) -> bool:
        return False

    @property
    def data_format(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def parse_files(self) -> pd.DataFrame:
        files = []
        folders = []
        file_paths = []
        for dir_obj in os.listdir(self.import_directory):
            if os.path.isdir(self.import_directory + '/' + dir_obj):
                for file in os.listdir(self.import_directory + '/' + dir_obj):
                    files.append(file)
                    folders.append(dir_obj)
                    file_paths.append(self.import_directory + '/' + dir_obj + '/' + file)
            elif os.path.isfile(self.import_directory + '/' + dir_obj):
                files.append(dir_obj)
                folders.append(self.import_directory.rpartition('/')[2])
                file_paths.append(self.import_directory + '/' + dir_obj)

        df = pd.DataFrame(files, columns=['file_names'])
        df['folders'] = folders
        df['file_paths'] = file_paths

        df = df[df['file_names'].str.contains(self.import_file_extension)]
        df['file_dates'] = df['file_names'].str.rpartition('_')[2].str.partition(self.import_file_extension)[0]
        df['file_dates'] = pd.to_datetime(df['file_dates'], format=self.import_file_date_format)
        return df

    @property
    def get_ingest_audit(self) -> pd.DataFrame:
        query = 'select max(ingest_datetime) from audit.ingest_load_times ' \
                + 'where schema_name = ' + self.schema \
                + ' and table_name = ' + self.table \
                + ' and job_name = ' + self.job_name
        return query

    def parse(self) -> pd.DataFrame:
        pass

    def execute(self):
        files = self.find_files_to_ingest
        df = self.data_format
        for idx, row in files.iterrows():
            raw = pd.read_csv(row['file_paths'])
            df = pd.concat([df, raw], sort=False)

        if bool(self.column_mapping):
            df = df.rename(columns=self.column_mapping)
            df = df[list(self.column_mapping.values())]

        if self.place_batch_file:
            df.to_csv(self.export_file_path(self.job_name), index=self.place_with_index)

        if self.load_to_db:
            if 'dw_created_at' not in df:
                ingest_datetime = datetime.datetime.now().utcnow().strftime("%m/%d/%Y %H:%M:%S")
                df['dw_created_at'] = ingest_datetime
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)

            insert_statement = 'INSERT INTO ' + self.schema + '.' + self.table + \
                ' (schema_name, table_name, job_name, ingest_datetime) ' + \
                ' VALUES ('
