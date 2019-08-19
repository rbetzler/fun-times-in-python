import datetime
from scripts.sql_scripts import table_creator
from scripts.sql_scripts.fred import ddls


class FredTableCreator(table_creator.TableCreator):
    @property
    def schema_name(self) -> str:
        return 'fred'


class FredReleasesTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'releases'

    @property
    def table_ddl(self) -> str:
        return ddls.RELEASES


class FredSeriesTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'series'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES


class FredSourcesTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'sources'

    @property
    def table_ddl(self) -> str:
        return ddls.SOURCES


if __name__ == '__main__':
    FredReleasesTable().execute()
    FredSeriesTable().execute()
    FredSourcesTable().execute()
