import abc
import datetime
import psycopg2
from finance.utilities import utils


class TableCreator(abc.ABC):

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

    def execute(self):
        conn = psycopg2.connect(self.db_connection)
        cursor = conn.cursor()

        print(f'Checking if schema exists: {datetime.datetime.utcnow()}')
        schema_check = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = '{self.schema_name}'
            )
            """
        cursor.execute(schema_check)
        schema_exists = cursor.fetchone()[0]
        if not schema_exists:
            print(f'Schema does not exist; creating now: {datetime.datetime.utcnow()}')
            cursor.execute(f'CREATE SCHEMA {self.schema_name};')
            conn.commit()

        if self.table_ddl:
            print(f'Running table ddl just in case: {datetime.datetime.utcnow()}')
            cursor.execute(self.table_ddl)
            conn.commit()

        if self.sql_script:
            print(f'Running sql script: {datetime.datetime.utcnow()}')
            cursor.execute(self.sql_script)
            conn.commit()

        conn.close()
        cursor.close()
