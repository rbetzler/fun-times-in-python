from finance.ingestion import table_creator
from finance.sql_scripts.internals import ddl


class IngestLoadTimesTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'audit'

    @property
    def table_name(self) -> str:
        return 'ingest_load_times'

    @property
    def table_ddl(self) -> str:
        return ddl.QUERY


if __name__ == '__main__':
    IngestLoadTimesTable().execute()
