import abc
import datetime
import psycopg2
from finance.utilities import utils


class SQLRunner(abc.ABC):

    @property
    def db_connection(self) -> str:
        return utils.DW_STOCKS

    @property
    @abc.abstractmethod
    def schema_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def table_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def table_ddl(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def sql_script(self) -> str:
        pass

    @property
    def run_maintenance(self) -> bool:
        """Run vacuum and analyze on table after sql scripts finish"""
        return True

    def execute(self):
        conn = psycopg2.connect(self.db_connection)
        cursor = conn.cursor()

        if self.schema_name:
            schema_query = f'''
                select exists (
                    select 1
                    from information_schema.schemata
                    where schema_name = '{self.schema_name}'
                )
                '''
            print(f'Checking if schema exists: {datetime.datetime.utcnow()}')
            cursor.execute(schema_query)
            schema_exists = cursor.fetchone()[0]
            if not schema_exists:
                print(f'Schema does not exist; creating now: {datetime.datetime.utcnow()}')
                cursor.execute(f'create schema {self.schema_name};')
                conn.commit()

        if self.table_name:
            table_query = f'''
                select exists (
                    select 1
                    from information_schema.tables
                    where table_schema = '{self.schema_name}'
                      and table_name = '{self.table_name}'
                )
                '''
            print(f'Checking if table exists: {datetime.datetime.utcnow()}')
            cursor.execute(table_query)
            table_exists = cursor.fetchone()[0]
            if not table_exists:
                print(f'Table does not exist; creating now: {datetime.datetime.utcnow()}')
                cursor.execute(self.table_ddl)
                conn.commit()

        if self.sql_script:
            print(f'Running sql script: {datetime.datetime.utcnow()}')
            cursor.execute(self.sql_script)
            conn.commit()

        cursor.close()
        conn.close()

        if self.run_maintenance:
            print('reconnecting to db to run maintenance scripts')
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor()
            conn.autocommit = True

            print('vacuuming table')
            cursor.execute(f'vacuum {self.schema_name}.{self.table_name};')

            print('analyzing table')
            cursor.execute(f'analyze {self.schema_name}.{self.table_name};')
            cursor.close()
            conn.close()
