from finance.data import sql
from finance.data.nasdaq import ddl


class NasdaqListedStocksTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'nasdaq'

    @property
    def table_name(self) -> str:
        return 'listed_stocks'

    @property
    def table_ddl(self) -> str:
        return ddl.QUERY


if __name__ == '__main__':
    NasdaqListedStocksTable().execute()
