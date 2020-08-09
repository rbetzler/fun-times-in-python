from finance.data import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'black_scholes'

    @property
    def directory(self) -> str:
        return 'reports/black_scholes'

    @property
    def import_file_prefix(self) -> str:
        return 'black_scholes_'

    @property
    def table(self) -> str:
        return 'black_scholes'

    @property
    def schema(self) -> str:
        return 'td'


if __name__ == '__main__':
    FileIngestion().execute()
