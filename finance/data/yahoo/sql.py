from finance.data import sql
from finance.data.yahoo import ddl


class Stocks(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'stocks'

    @property
    def table_ddl(self) -> str:
        return ddl.STOCKS

    @property
    def sql_script(self):
        return None


class IncomeStatements(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'income_statements'

    @property
    def table_ddl(self) -> str:
        return ddl.INCOME_STATEMENTS

    @property
    def sql_script(self):
        return None


class BalanceSheets(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'balance_sheets'

    @property
    def table_ddl(self) -> str:
        return ddl.BALANCE_SHEETS

    @property
    def sql_script(self):
        return None


class CashFlows(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'cash_flows'

    @property
    def table_ddl(self) -> str:
        return ddl.CASH_FLOWS

    @property
    def sql_script(self):
        return None


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

    @property
    def sql_script(self):
        return None


if __name__ == '__main__':
    Stocks().execute()
    IncomeStatements().execute()
    BalanceSheets().execute()
    CashFlows().execute()
    SPIndex().execute()
