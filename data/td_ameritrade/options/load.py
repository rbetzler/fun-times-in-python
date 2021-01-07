from data import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_options_api'

    @property
    def directory(self) -> str:
        return 'td_ameritrade/options'

    @property
    def import_file_prefix(self) -> str:
        return 'td_'

    @property
    def vacuum_analyze(self) -> bool:
        return False

    @property
    def schema(self) -> str:
        return 'td'

    @property
    def table(self) -> str:
        return 'options_raw'


if __name__ == '__main__':
    FileIngestion().execute()
