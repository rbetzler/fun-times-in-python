from finance.data import sql
from finance.data.internals import ddl


class IngestLoadTimesTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'audit'

    @property
    def table_name(self) -> str:
        return 'ingest_load_times'

    @property
    def table_ddl(self) -> str:
        return ddl.QUERY

    @property
    def sql_script(self) -> str:
        pass


if __name__ == '__main__':
    IngestLoadTimesTable().execute()
