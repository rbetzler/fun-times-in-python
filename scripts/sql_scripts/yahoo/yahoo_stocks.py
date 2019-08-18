import datetime
from scripts.sql_scripts import table_creator
from scripts.sql_scripts.yahoo import ddl


class YahooStocksTable(table_creator.TableCreator):

    @property
    def schema_name(self) -> str:
        return 'yahoo'

    @property
    def table_name(self) -> str:
        return 'stocks'

    @property
    def table_ddl(self) -> str:
        return ddl.QUERY


if __name__ == '__main__':
    YahooStocksTable().execute()
