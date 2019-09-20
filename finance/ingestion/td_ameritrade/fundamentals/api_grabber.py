import pandas as pd
from finance.ingestion import ingestion


class TDFundamentalsAPI(ingestion.Caller):
    # general
    @property
    def request_type(self) -> str:
        return 'api'

    @property
    def api_name(self) -> str:
        return 'API_TD'

    # calls
    @property
    def calls_query(self) -> str:
        query = """
            SELECT DISTINCT ticker
            FROM nasdaq.listed_stocks
            WHERE ticker !~ '[\^.~]'
            AND CHARACTER_LENGTH(ticker) BETWEEN 1 AND 4
            LIMIT {batch_size}
            OFFSET {batch_start}
            """
        return query.format(batch_size=self.batch_size, batch_start=self.lower_bound)

    def format_calls(self, idx, row) -> tuple:
        api_call = 'https://api.tdameritrade.com/v1/instruments' \
                   + '?apikey=' + self.api_secret \
                   + '&symbol=' + row.values[0] \
                   + '&projection=fundamental'
        api_name = row.values[0]
        return api_call, api_name

    # files
    @property
    def export_folder(self) -> str:
        folder = 'audit/td_ameritrade/fundamentals/' \
                 + self.folder_datetime \
                 + '/'
        return folder

    @property
    def export_file_name(self) -> str:
        return 'td_fundamentals_'

    @property
    def place_raw_file(self) -> bool:
        return True

    # parse
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
    batch_size = 10
    n_batches = 3
    for batch in range(1, n_batches):
        lower_bound = (batch-1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TDFundamentalsAPI(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
