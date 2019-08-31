from scripts.ingestion import table_creator
from scripts.ingestion.fred import ddls


class FredTableCreator(table_creator.TableCreator):
    @property
    def schema_name(self) -> str:
        return 'fred'


class FREDReleasesTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'releases'

    @property
    def table_ddl(self) -> str:
        return ddls.RELEASES


class FREDSeriesTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'series'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES


class FREDSourcesTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'sources'

    @property
    def table_ddl(self) -> str:
        return ddls.SOURCES


class FREDJobsTable(FredTableCreator):
    @property
    def table_name(self) -> str:
        return 'jobs'

    @property
    def table_ddl(self) -> str:
        return ddls.JOBS


if __name__ == '__main__':
    FREDReleasesTable().execute()
    FREDSeriesTable().execute()
    FREDSourcesTable().execute()
    FREDJobsTable().execute()
