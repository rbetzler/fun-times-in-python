from scripts.ingestion import table_creator
from scripts.ingestion.yahoo import ddl


class StocksTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'stocks'

    @property
    def table_ddl(self) -> str:
        return ddl.STOCKS


class IncomeStatementsTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'income_statements'

    @property
    def table_ddl(self) -> str:
        return ddl.INCOME_STATEMENTS


class BalanceSheetTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'balance_sheet'

    @property
    def table_ddl(self) -> str:
        return ddl.BALANCE_SHEET


if __name__ == '__main__':
    StocksTable().execute()
    IncomeStatementsTable().execute()
    BalanceSheetTable().execute()
