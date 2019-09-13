import pandas as pd
from finance.ingestion import file_ingestion


class ABCFREDSeriesFileIngestion(file_ingestion.FileIngestion):
    @property
    def import_directory(self) -> str:
        return 'audit/processed/fred/series'

    @property
    def schema(self) -> str:
        return 'fred'

    @property
    def load_to_db(self) -> bool:
        return True

    def clean_df(self, df) -> pd.DataFrame:
        df.loc[df['value'] == '.', 'value'] = 0
        return df


class FREDInflationFileIngestion(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_inflation'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_inflation_'

    @property
    def table(self) -> str:
        return 'inflation'


class FREDGovernmentDebtFileIngestion(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_central_government_debt'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_central_government_debt_'

    @property
    def table(self) -> str:
        return 'central_government_debt'


class FREDHouseholdDebtFileIngestion(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_household_debt_to_gdp'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_household_debt_to_gdp_'

    @property
    def table(self) -> str:
        return 'household_debt_to_gdp'


class FREDRealGDPPerCapitaFileIngestion(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_real_gdp_per_capita'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_real_gdp_per_capita_'

    @property
    def table(self) -> str:
        return 'real_gdp_per_capita'


class FREDStockMarketCapitalizationFileIngestion(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_stock_market_capitalization_to_gdp'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_stock_market_capitalization_to_gdp_'

    @property
    def table(self) -> str:
        return 'stock_market_capitalization_to_gdp'


if __name__ == '__main__':
    FREDInflationFileIngestion().execute()
    FREDGovernmentDebtFileIngestion().execute()
    FREDHouseholdDebtFileIngestion().execute()
    FREDRealGDPPerCapitaFileIngestion().execute()
    FREDStockMarketCapitalizationFileIngestion().execute()

