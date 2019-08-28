import time
import datetime
import pandas as pd
from scripts.ingestion import api_grabber
from scripts.sql_scripts.queries import td_option_tickers


class TdOptionsApi(api_grabber.ApiGrabber):
    @property
    def get_api_calls(self) -> pd.DataFrame:
        apis = []
        tickers = []
        for idx, row in self.tickers.iterrows():
            apis.append(self.api_call_base
                        + '?apikey=' + self.apikey
                        + '&symbol=' + row.values[0]
                        + '&contractType=' + self.contract_types)
            tickers.append(row.values[0])
        df = pd.DataFrame(data=apis, index=tickers)
        return df

    @property
    def contract_types(self) -> str:
        return 'ALL'

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
        return 'https://api.tdameritrade.com/v1/marketdata/chains'

    @property
    def apikey(self) -> str:
        return 'B41S3HBMUXQOLM81JXQ7CWXJMSN17CSM'

    @property
    def export_folder(self) -> str:
        folder = 'audit/processed/td_ameritrade/options/' \
                 + self.run_time.strftime('%Y_%m_%d_%H_%S') \
                 + '/'
        return folder

    @property
    def export_file_name(self) -> str:
        return 'td_'

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def n_workers(self) -> int:
        return 1

    @property
    def len_of_pause(self) -> int:
        return 1

    def column_renames(self) -> dict:
        names = {
            'putCall': 'put_call',
            'exchangeName': 'exchange_name',
            'bidSize': 'bid_size',
            'askSize': 'ask_size',
            'bidAskSize': 'bid_ask_size',
            'lastSize': 'last_size',
            'highPrice': 'high_price',
            'lowPrice': 'low_price',
            'openPrice': 'open_price',
            'closePrice': 'close_price',
            'totalVolume': 'total_volume',
            'tradeDate': 'trade_date',
            'tradeTimeInLong': 'trade_time_in_long',
            'quoteTimeInLong': 'quote_time_in_long',
            'netChange': 'net_change',
            'openInterest': 'open_interest',
            'timeValue': 'time_value',
            'theoreticalOptionValue': 'theoretical_option_value',
            'theoreticalVolatility': 'theoretical_volatility',
            'optionDeliverablesList': 'option_deliverable_list',
            'strikePrice': 'strike_price',
            'expirationDate': 'expiration_date',
            'daysToExpiration': 'days_to_expiration',
            'expirationType': 'expiration_type',
            'lastTradingDay': 'last_trading_day',
            'multiplier': 'multiplier',
            'settlementType': 'settlement_type',
            'deliverableNote': 'deliverable_note',
            'isIndexOption': 'is_index_option',
            'percentChange': 'percent_change',
            'markChange': 'mark_change',
            'markPercentChange': 'mark_percent_change',
            'nonStandard': 'non_standard',
            'inTheMoney': 'in_the_money'
        }
        return names

    def parse_chain(self, chain) -> pd.DataFrame:
        df = pd.DataFrame()
        print('Starting parse on chain: ' + datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        for date in chain.keys():
            for strike in chain.get(date).keys():
                temp = pd.DataFrame.from_dict(chain.get(date).get(strike))
                temp['expiration_date_from_epoch'] = \
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(temp['expirationDate'].values[0]/1000))
                temp['strike'] = strike
                temp['strike_date'] = date.partition(':')[0]
                temp['days_to_expiration_date'] = date.partition(':')[2]
                df = df.append(temp)

        df = df.rename(columns=self.column_renames())
        print('Ending parse on chain: ' + datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        return df

    def parse(self, res) -> pd.DataFrame:
        res = res.json()

        ticker = res.get('symbol')
        volatility = res.get('volatility')
        n_contracts = res.get('numberOfContracts')
        interest_rate = res.get('interestRate')
        calls = self.parse_chain(res.get('callExpDateMap'))
        puts = self.parse_chain(res.get('putExpDateMap'))

        df = calls.append(puts)
        df['symbol'] = ticker
        df['volatility'] = volatility
        df['n_contracts'] = n_contracts
        df['interest_rate'] = interest_rate
        return df


if __name__ == '__main__':
    batch_size = 100
    n_batches = 29
    for batch in range(1, n_batches):
        lower_bound = (batch-1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TdOptionsApi(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
