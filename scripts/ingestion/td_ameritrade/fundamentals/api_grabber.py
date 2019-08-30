import time
import datetime
import pandas as pd
from scripts.ingestion import api_grabber
from scripts.sql_scripts.queries import td_option_tickers


class TdFundamentalsApi(api_grabber.ApiGrabber):
    @property
    def get_api_calls(self) -> pd.DataFrame:
        apis = []
        tickers = []
        for idx, row in self.tickers.iterrows():
            apis.append(self.api_call_base
                        + '?apikey=' + self.apikey
                        + '&symbol=' + row.values[0]
                        + '&projection=' + self.search_type)
            tickers.append(row.values[0])
        df = pd.DataFrame(data=apis, index=tickers)
        return df

    @property
    def search_type(self) -> str:
        return 'fundamental'

    @property
    def query(self) -> str:
        return td_option_tickers.QUERY.format(
            batch_size=self.batch_size,
            batch_start=self.lower_bound
        )

    @property
    def tickers(self) -> pd.DataFrame:
        df = self.get_call_inputs_from_db
        return df

    @property
    def api_call_base(self) -> str:
        return 'https://api.tdameritrade.com/v1/instruments'

    @property
    def apikey(self) -> str:
        return 'B41S3HBMUXQOLM81JXQ7CWXJMSN17CSM'

    @property
    def export_folder(self) -> str:
        folder = 'audit/processed/td_ameritrade/fundamentals/' \
                 + self.run_time.strftime('%Y_%m_%d_%H_%S') \
                 + '/'
        return folder

    @property
    def export_file_name(self) -> str:
        return 'td_fundamentals_'

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def n_workers(self) -> int:
        return 15

    @property
    def len_of_pause(self) -> int:
        return 5

    @property
    def column_mapping(self) -> dict:
        names = {'high52': 'high_52',
                 'low52': 'low_52',
                 'dividendAmount': 'dividend_amount',
                 'dividendYield': 'dividend_yield',
                 'dividendDate': 'dividend_date',
                 'peRatio': 'pe_ratio',
                 'pegRatio': 'peg_ratio',
                 'pbRatio': 'pb_ratio',
                 'prRatio': 'pr_ratio',
                 'pcfRatio': 'pcf_ratio',
                 'grossMarginTTM': 'gross_margin_TTM',
                 'grossMarginMRQ': 'gross_margin_MRQ',
                 'netProfitMarginTTM': 'net_profit_margin_TTM',
                 'netProfitMarginMRQ': 'net_profit_margin_MRQ',
                 'operatingMarginTTM': 'operating_margin_TTM',
                 'operatingMarginMRQ': 'operating_margin_MRQ',
                 'returnOnEquity': 'return_on_equity',
                 'returnOnAssets': 'return_on_assets',
                 'returnOnInvestment': 'return_on_investment',
                 'quickRatio': 'quick_ratio',
                 'currentRatio': 'current_ratio',
                 'interestCoverage': 'interest_coverage',
                 'totalDebtToCapital': 'total_debt_to_capital',
                 'ltDebtToEquity': 'lt_debt_to_equity',
                 'totalDebtToEquity': 'total_debt_to_equity',
                 'epsTTM': 'eps_TTM',
                 'epsChangePercentTTM': 'eps_change_percent_TTM',
                 'epsChangeYear': 'eps_change_year',
                 'epsChange': 'eps_change',
                 'revChangeYear': 'rev_change_year',
                 'revChangeTTM': 'rev_change_TTM',
                 'revChangeIn': 'rev_change_in',
                 'sharesOutstanding': 'shares_outstanding',
                 'marketCapFloat': 'market_cap_float',
                 'marketCap': 'market_capitalization',
                 'bookValuePerShare': 'book_value_per_share',
                 'shortIntToFloat': 'short_int_to_float',
                 'shortIntDayToCover': 'short_int_day_to_cover',
                 'divGrowthRate3Year': 'div_growth_rate_3_year',
                 'dividendPayAmount': 'dividend_pay_amount',
                 'dividendPayDate': 'dividend_pay_date',
                 'vol1DayAvg': 'vol_1_day_average',
                 'vol10DayAvg': 'vol_10_day_average',
                 'vol3MonthAvg': 'vol_3_month_average'
                 }
        return names

    def parse(self, res) -> pd.DataFrame:
        res = res.json()

        df = pd.DataFrame()

        for key in res.keys():
            dictionary = res.get(key)

            temp = pd.DataFrame.from_dict(dictionary.get('fundamental'), orient='index').T
            temp['symbol'] = dictionary.get('symbol')
            temp['cusip'] = dictionary.get('cusip')
            temp['description'] = dictionary.get('description')
            temp['exchange'] = dictionary.get('exchange')
            temp['asset_type'] = dictionary.get('assetType')

            df = df.append(temp)

        return df


if __name__ == '__main__':
    batch_size = 100
    n_batches = 30
    for batch in range(1, n_batches):
        lower_bound = (batch-1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TdFundamentalsApi(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
