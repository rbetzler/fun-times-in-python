from finance.data import sql


class DevTradesSQLRunner(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        return 'dev'

    @property
    def table_name(self) -> str:
        return 'trades'

    @property
    def table_ddl(self) -> str:
        ddl = '''
        create table dev.trades (
              model_id                  varchar
            , market_datetime           date
            , symbol                    varchar
            , target                    numeric(20,6)
            , denormalized_target       numeric(20,6)
            , prediction                numeric(20,6)
            , denormalized_prediction   numeric(20,6)
            , normalization_min         numeric(20,6)
            , normalization_max         numeric(20,6)
            , file_datetime             timestamp
            , ingest_datetime           timestamp
        );
        '''
        return ddl

    @property
    def sql_script(self):
        return None


if __name__ == '__main__':
    DevTradesSQLRunner().execute()
