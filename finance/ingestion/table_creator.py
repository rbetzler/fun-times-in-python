import abc
import psycopg2
from finance.utilities import utils


class TableCreator(abc.ABC):

    @property
    def db_connection(self) -> str:
        return utils.DW_STOCKS

    @property
    def schema_name(self) -> str:
        pass

    @property
    def table_name(self) -> str:
        pass

    @property
    def table_ddl(self) -> str:
        pass

    @property
    def sql_script(self) -> str:
        pass

    def execute(self):
        conn = psycopg2.connect(self.db_connection)
        cursor = conn.cursor()

        schema_check = "SELECT EXISTS (SELECT 1 FROM information_schema.schemata " \
                       + "WHERE schema_name = '" + self.schema_name + "')"
        cursor.execute(schema_check)
        schema_exists = cursor.fetchone()[0]
        if not schema_exists:
            cursor.execute('CREATE SCHEMA ' + self.schema_name + ' ; ')
            conn.commit()

        if self.table_ddl:
            cursor.execute(self.table_ddl)
            conn.commit()

        if self.sql_script:
            cursor.execute(self.sql_script)
            conn.commit()

        conn.close()
        cursor.close()
