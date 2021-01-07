import abc

from data import sql
from data.fred import ddls


class FredSQLRunner(sql.SQLRunner, abc.ABC):
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

    @property
    def sql_script(self):
        return None


class FREDSeriesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'series'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES

    @property
    def sql_script(self):
        return None


class FREDSourcesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'sources'

    @property
    def table_ddl(self) -> str:
        return ddls.SOURCES

    @property
    def sql_script(self):
        return None


class FREDJobsTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'jobs'

    @property
    def table_ddl(self) -> str:
        return ddls.JOBS

    @property
    def sql_script(self):
        return None


class FREDSeriesSearchesTable(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return 'series_searches'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES_SEARCHES

    @property
    def sql_script(self):
        return None


class FREDSeriesTables(FredSQLRunner):
    @property
    def table_name(self) -> str:
        return '_series_template'

    @property
    def table_ddl(self) -> str:
        return ddls.SERIES_TEMPLATE

    @property
    def sql_script(self):
        return None


if __name__ == '__main__':
    FREDReleasesTable().execute()
    FREDSeriesTable().execute()
    FREDSourcesTable().execute()
    FREDJobsTable().execute()
    FREDSeriesSearchesTable().execute()
    FREDSeriesTables().execute()
