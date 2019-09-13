from finance.ingestion import table_creator
from finance.ingestion.td_ameritrade import ddls


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


class TdFundamentalsTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'fundamentals'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        return ddls.FUNDAMENTALS


if __name__ == '__main__':
    TdOptionsTableCreator().execute()
    TdEquitiesTableCreator().execute()
    TdFundamentalsTableCreator().execute()
