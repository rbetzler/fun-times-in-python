from finance.ingestion import table_creator
from finance.data_science.validation import ddl


class BenchmarkTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'benchmark'

    @property
    def schema_name(self) -> str:
        return 'validation'

    @property
    def table_ddl(self) -> str:
        return ddl.VALIDATION


if __name__ == '__main__':
    BenchmarkTableCreator().execute()
