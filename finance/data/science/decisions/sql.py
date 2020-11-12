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
              model_id                             varchar
            , decisioner_id                        varchar
            , model_datetime                       timestamp
            , market_datetime                      timestamp
            , symbol                               varchar
            , thirty_day_low_prediction            numeric(20,6)
            , close                                numeric(20,6)
            , put_call                             varchar
            , days_to_expiration                   numeric(20,6)
            , strike                               numeric(20,6)
            , price                                numeric(20,6)
            , potential_annual_return              numeric(20,6)
            , oom_percent                          numeric(20,6)
            , is_sufficiently_profitable           boolean
            , is_sufficiently_oom                  boolean
            , is_strike_below_predicted_low_price  boolean
            , quantity                             numeric(20,6)
            , asset                                varchar
            , direction                            varchar
            , file_datetime                        timestamp
            , ingest_datetime                      timestamp
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
