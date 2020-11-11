import abc
from finance.data import sql


class DecisionsSQLRunner(sql.SQLRunner, abc.ABC):
    @property
    def table_name(self) -> str:
        return 'decisions'

    @property
    def table_ddl(self) -> str:
        ddl = f'''
        create table {self.schema_name}.{self.table_name} (
              model_id                  varchar
            , model_datetime            timestamp
            , direction                 varchar
            , asset                     varchar
            , target                    numeric(20,6)
            , denormalized_target        numeric(20,6)
            , prediction                numeric(20,6)
            , denormalized_prediction    numeric(20,6)
            , quantity                  numeric(20,6)
            , symbol                    varchar
            , file_datetime             timestamp
            , ingest_datetime           timestamp
        );
        '''
        return ddl

    @property
    def sql_script(self):
        return None


class DevDecisionsSQLRunner(DecisionsSQLRunner):
    @property
    def schema_name(self) -> str:
        return 'dev'


class ProdDecisionsSQLRunner(DecisionsSQLRunner):
    @property
    def schema_name(self) -> str:
        return 'prod'


if __name__ == '__main__':
    DevDecisionsSQLRunner().execute()
    ProdDecisionsSQLRunner().execute()
