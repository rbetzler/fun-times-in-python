from data import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_quotes_api'

    @property
    def directory(self) -> str:
        return 'td_ameritrade/quotes'

    @property
    def import_file_prefix(self) -> str:
        return 'td_quotes_'

    @property
    def table(self) -> str:
        return 'quotes_raw'

    @property
    def schema(self) -> str:
        return 'td'


if __name__ == '__main__':
    FileIngestion().execute()
