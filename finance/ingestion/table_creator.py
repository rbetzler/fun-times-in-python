import abc
import psycopg2
from finance.utilities import utils
from finance.ingestion.yahoo import ddl


class TableCreator(abc.ABC):

    @property
    def db_connection(self) -> str:
        return utils.DW_STOCKS

    @property
    def schema_name(self) -> str:
        return ''

    @property
    def table_name(self) -> str:
        return ''

    @property
    def table_ddl(self) -> str:
        return ddl.ddl

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

        table_check = "SELECT EXISTS(SELECT 1 FROM information_schema.tables " \
                      + " WHERE table_schema = '" + self.schema_name + "'" \
                      + " AND table_name = '" + self.table_name + "') ;"
        cursor.execute(table_check)
        table_exists = cursor.fetchone()[0]
        if not table_exists:
            cursor.execute(self.table_ddl)
            conn.commit()

        conn.close()
        cursor.close()
