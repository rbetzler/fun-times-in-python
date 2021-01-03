from finance.data import sql


class IngestLoadTimesTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'audit'

    @property
    def table_name(self) -> str:
        return 'ingest_datetimes'

    @property
    def table_ddl(self) -> str:
        return '''
            create table if not exists audit.ingest_datetimes
            (
              schema_name       text,
              table_name        text,
              job_name          text,
              ingest_datetime   timestamp without time zone
            );
            '''

    @property
    def sql_script(self) -> str:
        pass


class HolidaysTable(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'utils'

    @property
    def table_name(self) -> str:
        return 'holidays'

    @property
    def table_ddl(self) -> str:
        return '''
            create table if not exists utils.holidays as (
              select '2021-01-01'::date as day_date, 'new_years_day' as holiday
              union
              select '2021-01-18', 'mlk_day'
              union
              select '2021-02-15', 'washingtons_bday'
              union
              select '2021-04-02', 'good_friday'
              union
              select '2021-05-31', 'memorial_day'
              union
              select '2021-07-05', 'independence_day'
              union
              select '2021-09-06', 'labor_day'
              union
              select '2021-11-25', 'thanksgiving'
              union
              select '2021-12-24', 'christmas'
            );
            '''

    @property
    def sql_script(self) -> str:
        pass


if __name__ == '__main__':
    IngestLoadTimesTable().execute()
    HolidaysTable().execute()
