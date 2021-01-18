from data import sql


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


if __name__ == '__main__':
    IngestLoadTimesTable().execute()
