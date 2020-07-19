from finance.data import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_options_api'

    @property
    def import_directory(self) -> str:
        return 'audit/td_ameritrade/options'

    @property
    def import_file_prefix(self) -> str:
        return 'td_'

    @property
    def export_folder(self) -> str:
        return 'audit/batches/td_ameritrade/options'

    @property
    def table(self) -> str:
        return 'options_raw'

    @property
    def schema(self) -> str:
        return 'td'


if __name__ == '__main__':
    FileIngestion().execute()
