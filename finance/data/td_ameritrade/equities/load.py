from finance.data import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_equities_api'

    @property
    def import_directory(self) -> str:
        return 'audit/td_ameritrade/equities'

    @property
    def import_file_prefix(self) -> str:
        return 'td_equities_'

    @property
    def export_folder(self) -> str:
        return 'audit/batches/td_ameritrade/equities'

    @property
    def table(self) -> str:
        return 'equities_raw'

    @property
    def schema(self) -> str:
        return 'td'


if __name__ == '__main__':
    FileIngestion().execute()
