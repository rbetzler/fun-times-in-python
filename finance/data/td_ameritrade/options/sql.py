from finance.data import sql


class TdOptionsSQLRunner(sql.SQLRunner):
    @property
    def table_name(self) -> str:
        return 'options'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = '''
            CREATE TABLE td.options (
                symbol                      text,
                volatility                  numeric(20,6),
                n_contracts                 numeric(20,6),
                interest_rate               numeric(20,6),
                put_call                    text,
                description                 text,
                exchange_name               text,
                bid                         numeric(20,6),
                ask                         numeric(20,6),
                last                        numeric(20,6),
                mark                        numeric(20,6),
                bid_size                    numeric(20,6),
                ask_size                    numeric(20,6),
                bid_ask_size                text,
                last_size                   numeric(20,6),
                high_price                  numeric(20,6),
                low_price                   numeric(20,6),
                open_price                  numeric(20,6),
                close_price                 numeric(20,6),
                total_volume                numeric(20,6),
                trade_date                  timestamp without time zone,
                trade_time_in_long          text,
                quote_time_in_long          text,
                net_change                  numeric(20,6),
                delta                       numeric(60,20),
                gamma                       numeric(60,20),
                theta                       numeric(60,20),
                vega                        numeric(60,20),
                rho                         numeric(60,20),
                open_interest               numeric(20,6),
                time_value                  numeric(20,6),
                theoretical_option_value    numeric(20,6),
                theoretical_volatility      numeric(20,6),
                option_deliverable_list     text,
                strike_price                numeric(20,6),
                expiration_date             text,
                days_to_expiration          numeric(20,6),
                expiration_type             text,
                last_trading_day            text,
                multiplier                  numeric(20,6),
                settlement_type             text,
                deliverable_note            text,
                is_index_option             text,
                percent_change              numeric(20,6),
                mark_change                 numeric(20,6),
                mark_percent_change         numeric(20,6),
                non_standard                text,
                mini                        text,
                in_the_money                text,
                expiration_date_from_epoch  timestamp without time zone,
                strike                      numeric(20,6),
                strike_date                 timestamp without time zone,
                days_to_expiration_date     numeric(20,6),
                file_datetime               timestamp without time zone,
                ingest_datetime             timestamp without time zone
            )
                PARTITION BY RANGE (file_datetime);

            CREATE TABLE IF NOT EXISTS td.options_2019_1 PARTITION OF td.options FOR VALUES FROM ('2019-1-01') TO ('2019-2-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_2 PARTITION OF td.options FOR VALUES FROM ('2019-2-01') TO ('2019-3-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_3 PARTITION OF td.options FOR VALUES FROM ('2019-3-01') TO ('2019-4-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_4 PARTITION OF td.options FOR VALUES FROM ('2019-4-01') TO ('2019-5-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_5 PARTITION OF td.options FOR VALUES FROM ('2019-5-01') TO ('2019-6-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_6 PARTITION OF td.options FOR VALUES FROM ('2019-6-01') TO ('2019-7-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_7 PARTITION OF td.options FOR VALUES FROM ('2019-7-01') TO ('2019-8-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_8 PARTITION OF td.options FOR VALUES FROM ('2019-8-01') TO ('2019-9-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_9 PARTITION OF td.options FOR VALUES FROM ('2019-9-01') TO ('2019-10-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_10 PARTITION OF td.options FOR VALUES FROM ('2019-10-01') TO ('2019-11-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_11 PARTITION OF td.options FOR VALUES FROM ('2019-11-01') TO ('2019-12-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_12 PARTITION OF td.options FOR VALUES FROM ('2019-12-01') TO ('2020-1-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_1 PARTITION OF td.options FOR VALUES FROM ('2020-1-01') TO ('2020-2-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_2 PARTITION OF td.options FOR VALUES FROM ('2020-2-01') TO ('2020-3-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_3 PARTITION OF td.options FOR VALUES FROM ('2020-3-01') TO ('2020-4-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_4 PARTITION OF td.options FOR VALUES FROM ('2020-4-01') TO ('2020-5-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_5 PARTITION OF td.options FOR VALUES FROM ('2020-5-01') TO ('2020-6-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_6 PARTITION OF td.options FOR VALUES FROM ('2020-6-01') TO ('2020-7-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_7 PARTITION OF td.options FOR VALUES FROM ('2020-7-01') TO ('2020-8-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_8 PARTITION OF td.options FOR VALUES FROM ('2020-8-01') TO ('2020-9-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_9 PARTITION OF td.options FOR VALUES FROM ('2020-9-01') TO ('2020-10-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_10 PARTITION OF td.options FOR VALUES FROM ('2020-10-01') TO ('2020-11-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_11 PARTITION OF td.options FOR VALUES FROM ('2020-11-01') TO ('2020-12-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_12 PARTITION OF td.options FOR VALUES FROM ('2020-12-01') TO ('2021-1-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_1 PARTITION OF td.options FOR VALUES FROM ('2021-1-01') TO ('2021-2-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_2 PARTITION OF td.options FOR VALUES FROM ('2021-2-01') TO ('2021-3-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_3 PARTITION OF td.options FOR VALUES FROM ('2021-3-01') TO ('2021-4-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_4 PARTITION OF td.options FOR VALUES FROM ('2021-4-01') TO ('2021-5-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_5 PARTITION OF td.options FOR VALUES FROM ('2021-5-01') TO ('2021-6-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_6 PARTITION OF td.options FOR VALUES FROM ('2021-6-01') TO ('2021-7-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_7 PARTITION OF td.options FOR VALUES FROM ('2021-7-01') TO ('2021-8-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_8 PARTITION OF td.options FOR VALUES FROM ('2021-8-01') TO ('2021-9-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_9 PARTITION OF td.options FOR VALUES FROM ('2021-9-01') TO ('2021-10-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_10 PARTITION OF td.options FOR VALUES FROM ('2021-10-01') TO ('2021-11-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_11 PARTITION OF td.options FOR VALUES FROM ('2021-11-01') TO ('2021-12-01');
            CREATE TABLE IF NOT EXISTS td.options_2021_12 PARTITION OF td.options FOR VALUES FROM ('2021-12-01') TO ('2022-1-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_1 PARTITION OF td.options FOR VALUES FROM ('2022-1-01') TO ('2022-2-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_2 PARTITION OF td.options FOR VALUES FROM ('2022-2-01') TO ('2022-3-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_3 PARTITION OF td.options FOR VALUES FROM ('2022-3-01') TO ('2022-4-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_4 PARTITION OF td.options FOR VALUES FROM ('2022-4-01') TO ('2022-5-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_5 PARTITION OF td.options FOR VALUES FROM ('2022-5-01') TO ('2022-6-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_6 PARTITION OF td.options FOR VALUES FROM ('2022-6-01') TO ('2022-7-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_7 PARTITION OF td.options FOR VALUES FROM ('2022-7-01') TO ('2022-8-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_8 PARTITION OF td.options FOR VALUES FROM ('2022-8-01') TO ('2022-9-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_9 PARTITION OF td.options FOR VALUES FROM ('2022-9-01') TO ('2022-10-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_10 PARTITION OF td.options FOR VALUES FROM ('2022-10-01') TO ('2022-11-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_11 PARTITION OF td.options FOR VALUES FROM ('2022-11-01') TO ('2022-12-01');
            CREATE TABLE IF NOT EXISTS td.options_2022_12 PARTITION OF td.options FOR VALUES FROM ('2022-12-01') TO ('2023-1-01');

            CREATE TABLE td.options_raw (
                symbol                      text,
                volatility                  numeric(20,6),
                n_contracts                 numeric(20,6),
                interest_rate               numeric(20,6),
                put_call                    text,
                description                 text,
                exchange_name               text,
                bid                         numeric(20,6),
                ask                         numeric(20,6),
                last                        numeric(20,6),
                mark                        numeric(20,6),
                bid_size                    numeric(20,6),
                ask_size                    numeric(20,6),
                bid_ask_size                text,
                last_size                   numeric(20,6),
                high_price                  numeric(20,6),
                low_price                   numeric(20,6),
                open_price                  numeric(20,6),
                close_price                 numeric(20,6),
                total_volume                numeric(20,6),
                trade_date                  timestamp without time zone,
                trade_time_in_long          text,
                quote_time_in_long          text,
                net_change                  numeric(20,6),
                delta                       numeric(60,20),
                gamma                       numeric(60,20),
                theta                       numeric(60,20),
                vega                        numeric(60,20),
                rho                         numeric(60,20),
                open_interest               numeric(20,6),
                time_value                  numeric(20,6),
                theoretical_option_value    numeric(20,6),
                theoretical_volatility      numeric(20,6),
                option_deliverable_list     text,
                strike_price                numeric(20,6),
                expiration_date             text,
                days_to_expiration          numeric(20,6),
                expiration_type             text,
                last_trading_day            text,
                multiplier                  numeric(20,6),
                settlement_type             text,
                deliverable_note            text,
                is_index_option             text,
                percent_change              numeric(20,6),
                mark_change                 numeric(20,6),
                mark_percent_change         numeric(20,6),
                non_standard                text,
                mini                        text,
                in_the_money                text,
                expiration_date_from_epoch  timestamp without time zone,
                strike                      numeric(20,6),
                strike_date                 timestamp without time zone,
                days_to_expiration_date     numeric(20,6),
                file_datetime               timestamp without time zone,
                ingest_datetime             timestamp without time zone
            )
                PARTITION BY RANGE (file_datetime);

            CREATE TABLE IF NOT EXISTS td.options_raw_2019_1 PARTITION OF td.options_raw FOR VALUES FROM ('2019-1-01') TO ('2019-2-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_2 PARTITION OF td.options_raw FOR VALUES FROM ('2019-2-01') TO ('2019-3-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_3 PARTITION OF td.options_raw FOR VALUES FROM ('2019-3-01') TO ('2019-4-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_4 PARTITION OF td.options_raw FOR VALUES FROM ('2019-4-01') TO ('2019-5-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_5 PARTITION OF td.options_raw FOR VALUES FROM ('2019-5-01') TO ('2019-6-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_6 PARTITION OF td.options_raw FOR VALUES FROM ('2019-6-01') TO ('2019-7-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_7 PARTITION OF td.options_raw FOR VALUES FROM ('2019-7-01') TO ('2019-8-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_8 PARTITION OF td.options_raw FOR VALUES FROM ('2019-8-01') TO ('2019-9-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_9 PARTITION OF td.options_raw FOR VALUES FROM ('2019-9-01') TO ('2019-10-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_10 PARTITION OF td.options_raw FOR VALUES FROM ('2019-10-01') TO ('2019-11-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_11 PARTITION OF td.options_raw FOR VALUES FROM ('2019-11-01') TO ('2019-12-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2019_12 PARTITION OF td.options_raw FOR VALUES FROM ('2019-12-01') TO ('2020-1-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_1 PARTITION OF td.options_raw FOR VALUES FROM ('2020-1-01') TO ('2020-2-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_2 PARTITION OF td.options_raw FOR VALUES FROM ('2020-2-01') TO ('2020-3-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_3 PARTITION OF td.options_raw FOR VALUES FROM ('2020-3-01') TO ('2020-4-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_4 PARTITION OF td.options_raw FOR VALUES FROM ('2020-4-01') TO ('2020-5-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_5 PARTITION OF td.options_raw FOR VALUES FROM ('2020-5-01') TO ('2020-6-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_6 PARTITION OF td.options_raw FOR VALUES FROM ('2020-6-01') TO ('2020-7-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_7 PARTITION OF td.options_raw FOR VALUES FROM ('2020-7-01') TO ('2020-8-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_8 PARTITION OF td.options_raw FOR VALUES FROM ('2020-8-01') TO ('2020-9-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_9 PARTITION OF td.options_raw FOR VALUES FROM ('2020-9-01') TO ('2020-10-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_10 PARTITION OF td.options_raw FOR VALUES FROM ('2020-10-01') TO ('2020-11-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_11 PARTITION OF td.options_raw FOR VALUES FROM ('2020-11-01') TO ('2020-12-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2020_12 PARTITION OF td.options_raw FOR VALUES FROM ('2020-12-01') TO ('2021-1-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_1 PARTITION OF td.options_raw FOR VALUES FROM ('2021-1-01') TO ('2021-2-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_2 PARTITION OF td.options_raw FOR VALUES FROM ('2021-2-01') TO ('2021-3-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_3 PARTITION OF td.options_raw FOR VALUES FROM ('2021-3-01') TO ('2021-4-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_4 PARTITION OF td.options_raw FOR VALUES FROM ('2021-4-01') TO ('2021-5-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_5 PARTITION OF td.options_raw FOR VALUES FROM ('2021-5-01') TO ('2021-6-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_6 PARTITION OF td.options_raw FOR VALUES FROM ('2021-6-01') TO ('2021-7-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_7 PARTITION OF td.options_raw FOR VALUES FROM ('2021-7-01') TO ('2021-8-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_8 PARTITION OF td.options_raw FOR VALUES FROM ('2021-8-01') TO ('2021-9-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_9 PARTITION OF td.options_raw FOR VALUES FROM ('2021-9-01') TO ('2021-10-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_10 PARTITION OF td.options_raw FOR VALUES FROM ('2021-10-01') TO ('2021-11-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_11 PARTITION OF td.options_raw FOR VALUES FROM ('2021-11-01') TO ('2021-12-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2021_12 PARTITION OF td.options_raw FOR VALUES FROM ('2021-12-01') TO ('2022-1-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_1 PARTITION OF td.options_raw FOR VALUES FROM ('2022-1-01') TO ('2022-2-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_2 PARTITION OF td.options_raw FOR VALUES FROM ('2022-2-01') TO ('2022-3-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_3 PARTITION OF td.options_raw FOR VALUES FROM ('2022-3-01') TO ('2022-4-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_4 PARTITION OF td.options_raw FOR VALUES FROM ('2022-4-01') TO ('2022-5-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_5 PARTITION OF td.options_raw FOR VALUES FROM ('2022-5-01') TO ('2022-6-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_6 PARTITION OF td.options_raw FOR VALUES FROM ('2022-6-01') TO ('2022-7-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_7 PARTITION OF td.options_raw FOR VALUES FROM ('2022-7-01') TO ('2022-8-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_8 PARTITION OF td.options_raw FOR VALUES FROM ('2022-8-01') TO ('2022-9-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_9 PARTITION OF td.options_raw FOR VALUES FROM ('2022-9-01') TO ('2022-10-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_10 PARTITION OF td.options_raw FOR VALUES FROM ('2022-10-01') TO ('2022-11-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_11 PARTITION OF td.options_raw FOR VALUES FROM ('2022-11-01') TO ('2022-12-01');
            CREATE TABLE IF NOT EXISTS td.options_raw_2022_12 PARTITION OF td.options_raw FOR VALUES FROM ('2022-12-01') TO ('2023-1-01');

            CREATE INDEX options_raw_symbol_file_ingest_idx ON td.options_raw (symbol, date(file_datetime), ingest_datetime DESC);
            '''
        return ddl

    @property
    def sql_script(self) -> str:
        script = '''
            drop index if exists td.options_symbol_idx;

            truncate td.options;
            insert into td.options (
                with
                dated as (
                    select *
                        , case
                            when extract(isodow from file_datetime) = 6 and extract('hour' from file_datetime) >= 20 then file_datetime::date + 3
                            when extract(isodow from file_datetime) = 7 then file_datetime::date + 2
                            when extract(isodow from file_datetime) = 1 or extract('hour' from file_datetime) >= 20 then file_datetime::date + 1
                            when extract(hour from file_datetime) < 13 or (extract('hour' from file_datetime) = 13 and extract('minute' from file_datetime) <= 30) then file_datetime::date
                            end as file_date
                    from td.options_raw
                    where file_datetime > current_date - 10 and file_datetime < current_date + 1
                    )
                , ranked as (
                    select *
                        , row_number() over(partition by file_date, symbol, put_call, strike, days_to_expiration order by ingest_datetime desc) as rn
                    from dated
                    where file_date is not null
                    )
                select
                    symbol
                    , volatility
                    , n_contracts
                    , interest_rate
                    , put_call
                    , description
                    , exchange_name
                    , bid
                    , ask
                    , last
                    , mark
                    , bid_size
                    , ask_size
                    , bid_ask_size
                    , last_size
                    , high_price
                    , low_price
                    , open_price
                    , close_price
                    , total_volume
                    , trade_date
                    , trade_time_in_long
                    , quote_time_in_long
                    , net_change
                    , delta
                    , gamma
                    , theta
                    , vega
                    , rho
                    , open_interest
                    , time_value
                    , theoretical_option_value
                    , theoretical_volatility
                    , option_deliverable_list
                    , strike_price
                    , expiration_date
                    , days_to_expiration
                    , expiration_type
                    , last_trading_day
                    , multiplier
                    , settlement_type
                    , deliverable_note
                    , is_index_option
                    , percent_change
                    , mark_change
                    , mark_percent_change
                    , non_standard
                    , mini
                    , in_the_money
                    , expiration_date_from_epoch
                    , strike
                    , strike_date
                    , days_to_expiration_date
                    , file_date as file_datetime
                    , ingest_datetime
                from ranked
                where rn = 1
                );

            create index if not exists options_symbol_idx on td.options (symbol);
            '''
        return script


if __name__ == '__main__':
    TdOptionsSQLRunner().execute()
