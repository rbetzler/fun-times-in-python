from finance.ingestion import table_creator
from finance.ingestion.yahoo import ddl


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


class SPIndex(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'sp_index'

    @property
    def table_ddl(self) -> str:
        return ddl.SP_INDEX


if __name__ == '__main__':
    StocksTable().execute()
    IncomeStatementsTable().execute()
    BalanceSheetsTable().execute()
    CashFlowsTable().execute()
    SPIndex().execute()
