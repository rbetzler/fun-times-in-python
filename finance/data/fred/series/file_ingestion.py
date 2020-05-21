import pandas as pd
from finance.data import loader


class ABCFREDSeriesFileIngestion(loader.FileIngestion):
    @property
    def import_directory(self) -> str:
        return 'audit/fred/series'

    @property
    def schema(self) -> str:
        return 'fred'

    @property
    def load_to_db(self) -> bool:
        return True

    def clean_df(self, df) -> pd.DataFrame:
        df.loc[df['value'] == '.', 'value'] = 0
        return df


class FREDInflation(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_inflation'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_inflation_'

    @property
    def table(self) -> str:
        return 'inflation'


class FREDGovernmentDebt(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_central_government_debt'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_central_government_debt_'

    @property
    def table(self) -> str:
        return 'central_government_debt'


class FREDHouseholdDebt(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_household_debt_to_gdp'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_household_debt_to_gdp_'

    @property
    def table(self) -> str:
        return 'household_debt_to_gdp'


class FREDRealGDPPerCapita(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_real_gdp_per_capita'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_real_gdp_per_capita_'

    @property
    def table(self) -> str:
        return 'real_gdp_per_capita'


class FREDStockMarketCapitalization(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_stock_market_capitalization_to_gdp'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_stock_market_capitalization_to_gdp_'

    @property
    def table(self) -> str:
        return 'stock_market_capitalization_to_gdp'


class FREDTreasuriesOneYear(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_1_year_treasury_bill'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_1_year_treasury_bill_'

    @property
    def table(self) -> str:
        return 'treasuries_one_year'


class FREDTreasuriesSixMonth(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_6_month_treasury_bill'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_6_month_treasury_bill_'

    @property
    def table(self) -> str:
        return 'treasuries_six_month'


class FREDTreasuriesThreeMonth(ABCFREDSeriesFileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_3_month_treasury_bill'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_3_month_treasury_bill_'

    @property
    def table(self) -> str:
        return 'treasuries_three_month'


if __name__ == '__main__':
    FREDInflation().execute()
    FREDGovernmentDebt().execute()
    FREDHouseholdDebt().execute()
    FREDRealGDPPerCapita().execute()
    FREDStockMarketCapitalization().execute()
    FREDTreasuriesOneYear().execute()
    FREDTreasuriesSixMonth().execute()
    FREDTreasuriesThreeMonth().execute()
