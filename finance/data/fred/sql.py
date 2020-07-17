from finance.data import sql
from finance.data.fred import ddls


class FredSQLRunner(sql.SQLRunner):
    @property
    def schema_name(self) -> str:
        return 'fred'


class FREDReleasesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'releases'

    @property
    def table_ddl(self) -> str:
        return ddls.RELEASES


class FREDSeriesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'series'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES


class FREDSourcesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'sources'

    @property
    def table_ddl(self) -> str:
        return ddls.SOURCES


class FREDJobsTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'jobs'

    @property
    def table_ddl(self) -> str:
        return ddls.JOBS


class FREDSeriesSearchesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'series_searches'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES_SEARCHES


class FREDSeriesTables(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return '_series_template'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES_TEMPLATE


if __name__ == '__main__':
    FREDReleasesTable().execute()
    FREDSeriesTable().execute()
    FREDSourcesTable().execute()
    FREDJobsTable().execute()
    FREDSeriesSearchesTable().execute()
    FREDSeriesTables().execute()
