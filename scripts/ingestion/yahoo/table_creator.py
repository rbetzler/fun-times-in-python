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


class BalanceSheetsTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'balance_sheets'

    @property
    def table_ddl(self) -> str:
        return ddl.BALANCE_SHEETS


class CashFlowsTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'cash_flows'

    @property
    def table_ddl(self) -> str:
        return ddl.CASH_FLOWS


if __name__ == '__main__':
    StocksTable().execute()
    IncomeStatementsTable().execute()
    BalanceSheetsTable().execute()
    CashFlowsTable().execute()
