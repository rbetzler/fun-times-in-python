from finance.ingestion import table_creator


class TdOptionsTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'options'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = """
            CREATE TABLE IF NOT EXISTS td.options (
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
            ) PARTITION BY RANGE (file_datetime);
            CREATE TABLE IF NOT EXISTS td.options_2019_1 PARTITION OF td.options 
                FOR VALUES FROM ('2019-01-01') TO ('2019-07-01');
            CREATE TABLE IF NOT EXISTS td.options_2019_2 PARTITION OF td.options 
                FOR VALUES FROM ('2019-07-01') TO ('2020-01-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_1 PARTITION OF td.options 
                FOR VALUES FROM ('2020-01-01') TO ('2020-07-01');
            CREATE TABLE IF NOT EXISTS td.options_2020_2 PARTITION OF td.options 
                FOR VALUES FROM ('2020-07-01') TO ('2021-01-01');
            CREATE TABLE IF NOT EXISTS td.options_detail (LIKE td.options);
            """
        return ddl

    @property
    def sql_script(self) -> str:
        script = """
        TRUNCATE td.options;
        INSERT INTO td.options (
            with partitioned as (
                select *, dense_rank() over(partition by symbol, date(file_datetime) order by ingest_datetime desc) as rn
                from td.options_detail
            )
            select 
                symbol,
                volatility,
                n_contracts,
                interest_rate,
                put_call,
                description,
                exchange_name,
                bid,
                ask,
                last,
                mark,
                bid_size,
                ask_size,
                bid_ask_size,
                last_size,
                high_price,
                low_price,
                open_price,
                close_price,
                total_volume,
                trade_date,
                trade_time_in_long,
                quote_time_in_long,
                net_change,
                delta,
                gamma,
                theta,
                vega,
                rho,
                open_interest,
                time_value,
                theoretical_option_value,
                theoretical_volatility,
                option_deliverable_list,
                strike_price,
                expiration_date,
                days_to_expiration,
                expiration_type,
                last_trading_day,
                multiplier,
                settlement_type,
                deliverable_note,
                is_index_option,
                percent_change,
                mark_change,
                mark_percent_change,
                non_standard,
                mini,
                in_the_money,
                expiration_date_from_epoch,
                strike,
                strike_date,
                days_to_expiration_date,
                file_datetime,
                ingest_datetime
            from partitioned
            where rn = 1);
        """
        return script


if __name__ == '__main__':
    TdOptionsTableCreator().execute()
