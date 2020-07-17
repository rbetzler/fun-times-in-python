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
    def is_maintenance(self) -> bool:
        """Maintenance scripts (like vacuums) require toggling autocommit on and off"""
        return False

    def execute(self):
        conn = psycopg2.connect(self.db_connection)
        cursor = conn.cursor()

        if self.schema_name:
            schema_query = f'''
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.schemata
                    WHERE schema_name = '{self.schema_name}'
                )
                '''
            print(f'Checking if schema exists: {datetime.datetime.utcnow()}')
            cursor.execute(schema_query)
            schema_exists = cursor.fetchone()[0]
            if not schema_exists:
                print(f'Schema does not exist; creating now: {datetime.datetime.utcnow()}')
                cursor.execute(f'CREATE SCHEMA {self.schema_name};')
                conn.commit()

        if self.table_name:
            table_query = f'''
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = '{self.table_name}'
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
            if self.is_maintenance:
                conn.autocommit = True
                cursor.execute(self.sql_script)
                conn.autocommit = False
            else:
                cursor.execute(self.sql_script)
                conn.commit()

        conn.close()
        cursor.close()
