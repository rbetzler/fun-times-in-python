from scripts.ingestion import table_creator
from scripts.ingestion.edgar import ddl


class EdgarFileTypesTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'edgar'

    @property
    def table_name(self) -> str:
        return 'file_types'

    @property
    def table_ddl(self) -> str:
        return ddl.QUERY


if __name__ == '__main__':
    EdgarFileTypesTable().execute()
