from scripts.ingestion import table_creator
from scripts.ingestion.td_ameritrade import ddls


class TdOptionsTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'options'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        return ddls.OPTIONS


class TdEquitiesTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'equities'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        return ddls.EQUITIES


if __name__ == '__main__':
    TdOptionsTableCreator().execute()
    TdEquitiesTableCreator().execute()
