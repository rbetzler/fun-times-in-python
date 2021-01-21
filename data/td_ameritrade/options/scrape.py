import time
import pandas as pd
from data import scraper


class TDOptionsAPI(scraper.Caller):
    @property
    def job_name(self) -> str:
        return 'API_TD'

    @property
    def calls_query(self) -> str:
        query = r'''
            select symbol as ticker
            from dbt.tickers
            order by 1
            limit {batch_size}
            offset {batch_start}
            '''
        return query.format(batch_size=self.batch_size, batch_start=self.lower_bound)

    def format_calls(self, row) -> tuple:
        key = row.ticker
        request = f'https://api.tdameritrade.com/v1/marketdata/chains?apikey={self.api_secret}&symbol={key}&contractType=ALL'
        return key, request

    @property
    def export_folder(self) -> str:
        return f'audit/td_ameritrade/options/{self.folder_datetime}/'

    @property
    def export_file_name(self) -> str:
        return 'td_options_'

    @property
    def n_workers(self) -> int:
        return 8

    @property
    def len_of_pause(self) -> int:
        return 5

    @property
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

    def parse_chain(self, chain: dict) -> pd.DataFrame:
        df = pd.DataFrame()
        for date in chain.keys():
            for strike in chain.get(date).keys():
                temp = pd.DataFrame.from_dict(chain.get(date).get(strike))
                temp['expiration_date_from_epoch'] = \
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(temp['expirationDate'].values[0]/1000))
                temp['strike'] = strike
                temp['strike_date'] = date.partition(':')[0]
                temp['days_to_expiration_date'] = date.partition(':')[2]
                df = df.append(temp)
        df = df.rename(columns=self.column_renames)
        return df

    def parse(self, res, key) -> pd.DataFrame:
        res = res.json()

        symbol = res.get('symbol')
        volatility = res.get('volatility')
        n_contracts = res.get('numberOfContracts')
        interest_rate = res.get('interestRate')
        calls = self.parse_chain(res.get('callExpDateMap'))
        puts = self.parse_chain(res.get('putExpDateMap'))

        df = calls.append(puts)
        df['symbol'] = symbol
        df['volatility'] = volatility
        df['n_contracts'] = n_contracts
        df['interest_rate'] = interest_rate
        return df


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch-1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TDOptionsAPI(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
