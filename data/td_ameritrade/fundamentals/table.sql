
create table if not exists td.fundamentals (
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
    gross_margin_ttm        numeric(20,6),
    gross_margin_mrq        numeric(20,6),
    net_profit_margin_ttm   numeric(20,6),
    net_profit_margin_mrq   numeric(20,6),
    operating_margin_ttm    numeric(20,6),
    operating_margin_mrq    numeric(20,6),
    return_on_equity        numeric(20,6),
    return_on_assets        numeric(20,6),
    return_on_investment    numeric(20,6),
    quick_ratio             numeric(20,6),
    current_ratio           numeric(20,6),
    interest_coverage       numeric(20,6),
    total_debt_to_capital   numeric(20,6),
    lt_debt_to_equity       numeric(20,6),
    total_debt_to_equity    numeric(20,6),
    eps_ttm                 numeric(20,6),
    eps_change_percent_ttm  numeric(20,6),
    eps_change_year         numeric(20,6),
    eps_change              numeric(20,6),
    rev_change_year         numeric(20,6),
    rev_change_ttm          numeric(20,6),
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
) partition by range (file_datetime);

create table if not exists td.fundamentals_2019_2 partition of td.fundamentals for values from ('2019-07-01') to ('2020-01-01');
create table if not exists td.fundamentals_2020_1 partition of td.fundamentals for values from ('2020-01-01') to ('2020-07-01');
create table if not exists td.fundamentals_2020_2 partition of td.fundamentals for values from ('2020-07-01') to ('2021-01-01');
create table if not exists td.fundamentals_2021_1 partition of td.fundamentals for values from ('2021-01-01') to ('2021-07-01');
