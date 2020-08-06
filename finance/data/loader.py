import os
import abc
import psycopg2
import datetime
import pandas as pd
from concurrent import futures
from sqlalchemy import create_engine, engine
from finance.utilities import utils

AUDIT_DIR = 'audit/'
BATCHES_DIR = 'batches/'


class FileIngestion(abc.ABC):
    def __init__(
            self,
            run_datetime=datetime.datetime.now(),
            n_files_to_process=0,
    ):
        self.db_connection = utils.DW_STOCKS
        self.run_datetime = run_datetime.strftime('%Y%m%d%H%M%S')
        self.ingest_datetime = run_datetime.strftime("%Y-%m-%d %H:%M:%S")
        self.n_files_to_process = n_files_to_process
        self.n_workers = None

    @property
    @abc.abstractmethod
    def job_name(self) -> str:
        pass

    @property
    def quote_character(self) -> str:
        return '"'

    @property
    @abc.abstractmethod
    def directory(self) -> str:
        pass

    @property
    def import_directory(self):
        return AUDIT_DIR + self.directory

    @property
    @abc.abstractmethod
    def import_file_prefix(self) -> str:
        pass

    @property
    def import_file_extension(self) -> str:
        return '.csv'

    @property
    def import_file_date_format(self) -> str:
        return '%Y%m%d%H%M%S'

    @property
    def export_folder(self) -> str:
        return AUDIT_DIR + BATCHES_DIR + self.directory

    @property
    def export_file_type(self) -> str:
        return '.csv'

    def export_file_path(self, job) -> str:
        file_path = f'{self.export_folder}/batch_{job}_{self.run_datetime}{self.export_file_type}'
        return file_path

    @property
    def export_file_separator(self) -> str:
        return ','

    @property
    @abc.abstractmethod
    def schema(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def table(self) -> str:
        pass

    @property
    def db_engine(self) -> engine.Engine:
        return create_engine(self.db_connection)

    @property
    def header_row(self) -> bool:
        return False

    @property
    def column_mapping(self) -> dict:
        return {}

    @property
    def get_columns_in_db(self) -> list:
        query = f'select * from {self.schema}.{self.table} limit 1'
        df = utils.query_db(query=query)
        cols = list(df.columns)
        return cols

    def add_and_order_columns(self, df) -> pd.DataFrame:
        cols = self.get_columns_in_db
        for col in cols:
            if col not in df:
                df[col] = None
        df = df[cols]
        return df

    def clean_df(self, df) -> pd.DataFrame:
        return df

    @property
    def get_available_files(self) -> pd.DataFrame:
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
        df = df[df['file_names'].str[:len(self.import_file_prefix)] == self.import_file_prefix]

        df['file_dates'] = df['file_names'].str.rpartition('_')[2].str.partition(self.import_file_extension)[0]
        df['file_dates'] = pd.to_datetime(df['file_dates'], format=self.import_file_date_format)
        df['file_modified_datetime'] = df['file_paths'].apply(os.path.getmtime)
        df['file_modified_datetime'] = pd.to_datetime(df['file_modified_datetime'], unit='s')
        return df

    @property
    def get_ingest_audit(self) -> pd.DataFrame:
        query = f'select 1 from {self.schema}.{self.table} limit 1'
        df = utils.query_db(query=query)
        if not df.empty:
            query = f'''
                select coalesce(max(ingest_datetime), '1900-01-01') as ingest_datetime
                from audit.ingest_datetimes
                where   schema_name = '{self.schema}'
                    and table_name = '{self.table}'
                    and job_name = '{self.job_name}'
                '''
            df = utils.query_db(query=query)
            ingest_datetime = df['ingest_datetime'].values[0]
        else:
            ingest_datetime = '2000-01-01'
        return ingest_datetime

    @property
    def get_ingest_files(self) -> pd.DataFrame:
        available_files = self.get_available_files
        last_ingest_datetime = self.get_ingest_audit
        df = available_files[available_files['file_modified_datetime'] >= last_ingest_datetime]
        df = df.sort_values(by=['file_modified_datetime'])

        print(f'{len(df)} files need to be ingested')
        if self.n_files_to_process > 0:
            file_modified_datetime = df.loc[self.n_files_to_process - 1, 'file_modified_datetime']
            df = df[df['file_modified_datetime'] <= file_modified_datetime]

        print(f'Will ingest {len(df)} files')
        return df

    @staticmethod
    def get_files(file_path, file_datetime):
        if os.stat(file_path).st_size > 1:
            raw = pd.read_csv(file_path)
            raw['file_datetime'] = file_datetime
            return raw

    def insert_audit_record(self, ingest_datetime: str):
        query = f'''
            INSERT INTO audit.ingest_datetimes
            (schema_name, table_name, job_name, ingest_datetime)
            VALUES ('{self.schema}', '{self.table}', '{self.job_name}', '{ingest_datetime}')
            '''
        utils.insert_record(query=query)
        return

    def execute(self):
        print('Getting list of files to ingest')
        files = self.get_ingest_files

        if not files.empty:
            executor = futures.ProcessPoolExecutor(max_workers=self.n_workers)
            future_to_url = {
                executor.submit(
                    self.get_files,
                    row['file_paths'],
                    row['file_dates'],
                ): row for _, row in files.iterrows()}

            raw_dfs = []
            for future in futures.as_completed(future_to_url):
                if future.result() is not None:
                    raw_dfs.append(future.result())

            df = pd.concat(raw_dfs, sort=False)

            if not df.empty:
                df = self.clean_df(df)
                if bool(self.column_mapping):
                    print('renaming columns')
                    df = df.rename(columns=self.column_mapping)

                if 'ingest_datetime' not in df:
                    print('adding ingest_datetime')
                    df['ingest_datetime'] = self.ingest_datetime

                df = self.add_and_order_columns(df)

                print('placing batch file')
                df.to_csv(
                    self.export_file_path(self.job_name),
                    index=False,
                    header=self.header_row,
                    sep=self.export_file_separator,
                )
                file = open(self.export_file_path(self.job_name), 'r')

                print('copying to db')
                conn = psycopg2.connect(self.db_connection)
                cursor = conn.cursor()
                copy_command = f'''
                    COPY {self.schema}.{self.table}
                    FROM STDIN
                    DELIMITER '{self.export_file_separator}' QUOTE '{self.quote_character}' CSV
                    '''
                cursor.copy_expert(copy_command, file=open(self.export_file_path(self.job_name)))
                conn.commit()

                print('vacuuming table')
                conn.autocommit = True
                cursor.execute(f'vacuum {self.schema}.{self.table};')

                print('analyzing table')
                cursor.execute(f'analyze {self.schema}.{self.table};')

                print('closing db connection')
                cursor.close()
                conn.close()
                file.close()

                self.insert_audit_record(ingest_datetime=files['file_modified_datetime'].max())

        else:
            print('no files to ingest')
