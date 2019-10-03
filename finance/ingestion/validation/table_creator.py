from finance.ingestion import table_creator
from finance.ingestion.validation import sql_script


class BenchmarkTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'benchmark'

    @property
    def schema_name(self) -> str:
        return 'validation'

    @property
    def table_ddl(self) -> str:
        ddl = """
            CREATE TABLE IF NOT EXISTS validation.benchmarks
            (
              market_datetime   timestamp without time zone,
              model_type        text,
              profit_loss       numeric(20,6),
              ingest_datetime   timestamp without time zone
            );
            """
        return ddl

    @property
    def sql_script(self) -> str:
        return sql_script.MOVING_AVERAGE


if __name__ == '__main__':
    BenchmarkTableCreator().execute()
