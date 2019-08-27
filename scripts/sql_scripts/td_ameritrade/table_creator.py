from scripts.sql_scripts import table_creator
from scripts.sql_scripts.td_ameritrade import ddls


class TdOptionsTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'options'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        return ddls.QUERY


if __name__ == '__main__':
    TdOptionsTableCreator().execute()
