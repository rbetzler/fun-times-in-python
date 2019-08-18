import datetime
from scripts.sql_scripts import table_creator
from scripts.sql_scripts.edgar import ddl


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
