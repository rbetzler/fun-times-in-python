from finance.data import sql
from finance.data.yahoo import ddl


class StocksTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'stocks'

    @property
    def table_ddl(self) -> str:
        return ddl.STOCKS


class IncomeStatementsTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'income_statements'

    @property
    def table_ddl(self) -> str:
        return ddl.INCOME_STATEMENTS


class BalanceSheetsTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'balance_sheets'

    @property
    def table_ddl(self) -> str:
        return ddl.BALANCE_SHEETS


class CashFlowsTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'cash_flows'

    @property
    def table_ddl(self) -> str:
        return ddl.CASH_FLOWS


class SPIndex(sql.SQLRunner):

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
