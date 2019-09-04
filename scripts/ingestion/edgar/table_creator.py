from scripts.ingestion import table_creator
from scripts.ingestion.edgar import ddls


class ABCEdgarTable(table_creator.TableCreator):
    @property
    def schema_name(self) -> str:
        return 'edgar'


class FileTypesTable(ABCEdgarTable):
    @property
    def table_name(self) -> str:
        return 'file_types'

    @property
    def table_ddl(self) -> str:
        return ddls.FILE_TYPES


class SICCIKCodesTable(ABCEdgarTable):
    @property
    def table_name(self) -> str:
        return 'sic_cik_codes'

    @property
    def table_ddl(self) -> str:
        return ddls.SIC_CIK_CODES


class FilingsTable(ABCEdgarTable):
    @property
    def table_name(self) -> str:
        return 'filings'

    @property
    def table_ddl(self) -> str:
        return ddls.FILINGS


if __name__ == '__main__':
    FileTypesTable().execute()
    SICCIKCodesTable().execute()
    FilingsTable().execute()
