import datetime
import pandas as pd
from data import loader


class ListStocks(loader.FileIngestion):
    @property
    def input_folder(self) -> str:
        return '/nasdaq/'

    @property
    def input_file_name(self) -> str:
        file_in = 'companylist_' + self.exchange + '_2019_08_18'
        return file_in

    # overriding since this was manually extracted
    @property
    def input_file_path(self) -> str:
        file_path = 'audit' \
                    + self.input_folder \
                    + self.input_file_name \
                    + self.input_file_type
        return file_path

    @property
    def input_file_type(self) -> str:
        return '.csv'

    @property
    def exchange(self) -> str:
        return ''

    @property
    def export_folder(self) -> str:
        return '/' + self.exchange + '/'

    @property
    def export_file_name(self) -> str:
        return 'listed_stocks_' + self.exchange + '.py'

    @property
    def export_file_type(self) -> str:
        return '.csv'

    @property
    def export_file_path(self) -> str:
        file_path = 'audit/processed' \
                    + self.export_folder \
                    + self.export_file_name \
                    + self.run_date \
                    + self.export_file_type
        return file_path

    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        return 'listed_stocks'

    @property
    def schema(self) -> str:
        return 'nasdaq'

    @property
    def column_mapping(self) -> dict:
        cols = {
            'Symbol': 'ticker',
            'ADR TSO': 'adr_tso',
            'Name': 'company',
            'IPOyear': 'ipo_year',
            'Sector': 'sector',
            'Industry': 'industry'
        }
        return cols

    def parse(self, file) -> pd.DataFrame:
        df = file.rename(columns=self.column_mapping)
        df = df[list(self.column_mapping.values())]
        df['exchange'] = self.exchange
        return df

    def execute(self):
        file = pd.read_csv(self.input_file_path)
        df = self.parse(file)

        if self.place_raw_file:
            df.to_csv(self.export_file_path, index=self.place_with_index)

        if self.load_to_db:
            if 'dw_created_at' not in df:
                df['dw_created_at'] = datetime.datetime.now().utcnow().strftime("%m/%d/%Y %H:%M:%S")
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=False)


class AmexListStocks(ListStocks):
    @property
    def exchange(self) -> str:
        return 'amex'


class NasdaqListStocks(ListStocks):
    @property
    def exchange(self) -> str:
        return 'nasdaq'


class NyseListStocks(ListStocks):
    @property
    def exchange(self) -> str:
        return 'nyse'


if __name__ == '__main__':
    AmexListStocks().execute()
    NasdaqListStocks().execute()
    NyseListStocks().execute()
