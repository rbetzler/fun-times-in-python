from scripts.ingestion import table_creator
from scripts.ingestion.nasdaq import ddl


class NasdaqListedStocksTable(table_creator.TableCreator):

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
