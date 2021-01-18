from data import sql


class TDQuotesSQLRunner(sql.SQLRunner):
    @property
    def table_name(self) -> str:
        return 'quotes_raw'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = '''
            create table td.quotes_raw (
                asset_type varchar
              , asset_main_type varchar
              , cusip varchar
              , assetSubType varchar
              , symbol varchar
              , description varchar
              , bid_price numeric(12,2)
              , bid_size integer
              , bid_id varchar
              , ask_price numeric(12,2)
              , ask_size integer
              , ask_id varchar
              , last_price numeric(12,2)
              , last_size integer
              , last_id varchar
              , open_price numeric(12,2)
              , high_price numeric(12,2)
              , low_price numeric(12,2)
              , bid_tick varchar
              , close_price numeric(12,2)
              , net_change numeric(12,2)
              , total_volume integer
              , quote_time_in_long bigint
              , trade_time_in_long bigint
              , mark numeric(12,2)
              , exchange varchar
              , exchange_name varchar
              , marginable boolean
              , shortable boolean
              , volatility numeric(12,2)
              , digits varchar
              , "52_week_high" numeric(12,2)
              , "52_week_low" numeric(12,2)
              , nav numeric(12,2)
              , pe_ratio numeric(12,2)
              , dividend_amount numeric(12,2)
              , dividend_yield numeric(12,2)
              , dividend_date timestamp
              , security_status varchar
              , regular_market_last_price numeric(12,2)
              , regular_market_last_size integer
              , regular_market_net_change numeric(12,2)
              , regular_market_trade_time_in_long bigint
              , net_percent_change_in_double numeric(12,2)
              , mark_change_in_double numeric(12,2)
              , mark_percent_change_in_double numeric(12,2)
              , regular_market_percent_change_in_double numeric(12,2)
              , delayed boolean
              , quote_time_in_long_datetime timestamp
              , trade_time_in_long_datetime timestamp
              , regular_market_trade_time_in_long_datetime timestamp
              , file_datetime timestamp without time zone
              , ingest_datetime timestamp without time zone
            )
                partition by range (file_datetime);

            create table td.quotes_raw_20208 partition of td.quotes_raw for values from ('2020-08-01') to ('2020-09-01');
            create table td.quotes_raw_20209 partition of td.quotes_raw for values from ('2020-09-01') to ('2020-10-01');
            create table td.quotes_raw_202010 partition of td.quotes_raw for values from ('2020-10-01') to ('2020-11-01');
            create table td.quotes_raw_202011 partition of td.quotes_raw for values from ('2020-11-01') to ('2020-12-01');
            create table td.quotes_raw_202012 partition of td.quotes_raw for values from ('2020-12-01') to ('2021-01-01');
            create table td.quotes_raw_20211 partition of td.quotes_raw for values from ('2021-01-01') to ('2021-02-01');
            create table td.quotes_raw_20212 partition of td.quotes_raw for values from ('2021-02-01') to ('2021-03-01');
            create table td.quotes_raw_20213 partition of td.quotes_raw for values from ('2021-03-01') to ('2021-04-01');
            create table td.quotes_raw_20214 partition of td.quotes_raw for values from ('2021-04-01') to ('2021-05-01');
            create table td.quotes_raw_20215 partition of td.quotes_raw for values from ('2021-05-01') to ('2021-06-01');
            create table td.quotes_raw_20216 partition of td.quotes_raw for values from ('2021-06-01') to ('2021-07-01');
            create table td.quotes_raw_20217 partition of td.quotes_raw for values from ('2021-07-01') to ('2021-08-01');
            create table td.quotes_raw_20218 partition of td.quotes_raw for values from ('2021-08-01') to ('2021-09-01');
            create table td.quotes_raw_20219 partition of td.quotes_raw for values from ('2021-09-01') to ('2021-10-01');
            create table td.quotes_raw_202110 partition of td.quotes_raw for values from ('2021-10-01') to ('2021-11-01');
            create table td.quotes_raw_202111 partition of td.quotes_raw for values from ('2021-11-01') to ('2021-12-01');
            '''
        return ddl


if __name__ == '__main__':
    TDQuotesSQLRunner().execute()
