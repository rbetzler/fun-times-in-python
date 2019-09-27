OPTIONS = """
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
    -- CREATE INDEX ON td.options (symbol);
    CREATE OR REPLACE VIEW td.options_view AS (
        with partitioned as (
            select *, row_number() over(partition by symbol, date(file_datetime) order by ingest_datetime desc) as rn
            from td.options
        )
        select *
        from partitioned
        where rn = 1);
    """

EQUITIES = """
    CREATE TABLE IF NOT EXISTS td.equities (
        symbol                  text,
        open                    numeric(20,6),
        high                    numeric(20,6),
        low                     numeric(20,6),
        close                   numeric(20,6),
        volume                  numeric(20,6),
        market_datetime_epoch   text,
        empty                   boolean,
        market_datetime         timestamp without time zone,
        file_datetime           timestamp without time zone,
        ingest_datetime         timestamp without time zone
    ) PARTITION BY RANGE (market_datetime);
    CREATE TABLE IF NOT EXISTS td.equities_1990 PARTITION OF td.equities 
        FOR VALUES FROM ('1990-01-01') TO ('1995-01-01');
    CREATE TABLE IF NOT EXISTS td.equities_1995 PARTITION OF td.equities 
        FOR VALUES FROM ('1995-01-01') TO ('2000-01-01');
    CREATE TABLE IF NOT EXISTS td.equities_2000 PARTITION OF td.equities 
        FOR VALUES FROM ('2000-01-01') TO ('2005-01-01');
    CREATE TABLE IF NOT EXISTS td.equities_2005 PARTITION OF td.equities 
        FOR VALUES FROM ('2005-01-01') TO ('2010-01-01');
    CREATE TABLE IF NOT EXISTS td.equities_2010 PARTITION OF td.equities 
        FOR VALUES FROM ('2010-01-01') TO ('2015-01-01');
    CREATE TABLE IF NOT EXISTS td.equities_2015 PARTITION OF td.equities 
        FOR VALUES FROM ('2015-01-01') TO ('2020-01-01');    
    CREATE TABLE IF NOT EXISTS td.equities_2020 PARTITION OF td.equities 
        FOR VALUES FROM ('2020-01-01') TO ('2025-01-01');
    -- CREATE INDEX ON td.equities (symbol);
    CREATE OR REPLACE VIEW td.equities_view AS (
        with partitioned as (
            select *, row_number() over(partition by symbol, market_datetime order by file_datetime desc) as rn
            from td.equities
        )
        select *
        from partitioned
        where rn = 1);
    """

FUNDAMENTALS = """
    CREATE TABLE IF NOT EXISTS td.fundamentals (
        symbol                  text,
        high_52                 numeric(20,6),
        low_52                  numeric(20,6),
        dividend_amount         numeric(20,6),
        dividend_yield          numeric(20,6),
        dividend_date           timestamp without time zone,
        pe_ratio                numeric(20,6),
        peg_ratio               numeric(20,6),
        pb_ratio                numeric(20,6),
        pr_ratio                numeric(20,6),
        pcf_ratio               numeric(20,6),
        gross_margin_TTM        numeric(20,6),
        gross_margin_MRQ        numeric(20,6),
        net_profit_margin_TTM   numeric(20,6),
        net_profit_margin_MRQ   numeric(20,6),
        operating_margin_TTM    numeric(20,6),
        operating_margin_MRQ    numeric(20,6),
        return_on_equity        numeric(20,6),
        return_on_assets        numeric(20,6),
        return_on_investment    numeric(20,6),
        quick_ratio             numeric(20,6),
        current_ratio           numeric(20,6),
        interest_coverage       numeric(20,6),
        total_debt_to_capital   numeric(20,6),
        lt_debt_to_equity       numeric(20,6),
        total_debt_to_equity    numeric(20,6),
        eps_TTM                 numeric(20,6),
        eps_change_percent_TTM  numeric(20,6),
        eps_change_year         numeric(20,6),
        eps_change              numeric(20,6),
        rev_change_year         numeric(20,6),
        rev_change_TTM          numeric(20,6),
        rev_change_in           numeric(20,6),
        shares_outstanding      numeric(20,6),
        market_cap_float        numeric(20,6),
        market_capitalization   numeric(20,6),
        book_value_per_share    numeric(20,6),
        short_int_to_float      numeric(20,6),
        short_int_day_to_cover  numeric(20,6),
        div_growth_rate_3_year  numeric(20,6),
        dividend_pay_amount     numeric(20,6),
        dividend_pay_date       timestamp without time zone,
        vol_1_day_average       numeric(20,6),
        vol_10_day_average      numeric(20,6),
        vol_3_month_average     numeric(20,6),
        cusip                   text,
        description             text,
        exchange                text,
        asset_type              text,
        file_datetime           timestamp without time zone,
        ingest_datetime         timestamp without time zone
    ) PARTITION BY RANGE (file_datetime);
    -- CREATE INDEX ON td.fundamentals (symbol);
    CREATE TABLE IF NOT EXISTS td.fundamentals_2019 PARTITION OF td.fundamentals 
        FOR VALUES FROM ('2019-01-01') TO ('2020-01-01');
    """