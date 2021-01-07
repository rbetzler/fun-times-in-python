import abc
from data import sql


class PredictionsSQLRunner(sql.SQLRunner, abc.ABC):
    @property
    def table_name(self) -> str:
        return 'predictions'

    @property
    def table_ddl(self) -> str:
        ddl = f'''
        create table {self.schema_name}.{self.table_name} (
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


class DevPredictionsSQLRunner(PredictionsSQLRunner):
    @property
    def schema_name(self) -> str:
        return 'dev'


class ProdPredictionsSQLRunner(PredictionsSQLRunner):
    @property
    def schema_name(self) -> str:
        return 'prod'


if __name__ == '__main__':
    DevPredictionsSQLRunner().execute()
    ProdPredictionsSQLRunner().execute()
