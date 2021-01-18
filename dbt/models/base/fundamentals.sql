{{
  config(
    materialized='table',
    post_hook='''
      create index if not exists {{ this.name }}_symbol_idx on {{ this }} (symbol);
      create index if not exists {{ this.name }}_market_datetime_idx on {{ this }} (market_datetime);
      '''
  )
}}

with
raw as (
  select *
    , case when h.day_date is not null then f.file_datetime - interval '1 days' else f.file_datetime end as offset_file_datetime
  from {{ source('td', 'fundamentals') }} as f
  left join {{ ref('holidays') }} as h
    on  f.file_datetime::date = h.day_date
)
, dated as (
  select *
    , case
      -- If Sunday, set to Friday
      when extract(isodow from offset_file_datetime) = 7 then offset_file_datetime::date - 2
      -- If Saturday, set to Friday
      when extract(isodow from offset_file_datetime) = 6 then offset_file_datetime::date - 1
      -- If after market close, leave as is
      when extract('hour' from offset_file_datetime) >= 20 then offset_file_datetime::date
      -- If before market open, set to prior day
      when extract(hour from offset_file_datetime) < 13 or (extract('hour' from offset_file_datetime) = 13 and extract('minute' from offset_file_datetime) <= 30) then offset_file_datetime::date - 1
      end as file_date
  from raw
)
, ranked as (
  select *
    , row_number() over(partition by file_date, symbol order by ingest_datetime desc) as rn
  from dated
  where file_date is not null
)
select
    symbol
  , high_52
  , low_52
  , dividend_amount
  , dividend_yield
  , dividend_date
  , pe_ratio
  , peg_ratio
  , pb_ratio
  , pr_ratio
  , pcf_ratio
  , gross_margin_ttm
  , gross_margin_mrq
  , net_profit_margin_ttm
  , net_profit_margin_mrq
  , operating_margin_ttm
  , operating_margin_mrq
  , return_on_equity
  , return_on_assets
  , return_on_investment
  , quick_ratio
  , current_ratio
  , interest_coverage
  , total_debt_to_capital
  , lt_debt_to_equity
  , total_debt_to_equity
  , eps_ttm
  , eps_change_percent_ttm
  , eps_change_year
  , eps_change
  , rev_change_year
  , rev_change_ttm
  , rev_change_in
  , shares_outstanding
  , market_cap_float
  , market_capitalization
  , book_value_per_share
  , short_int_to_float
  , short_int_day_to_cover
  , div_growth_rate_3_year
  , dividend_pay_amount
  , dividend_pay_date
  , vol_1_day_average
  , vol_10_day_average
  , vol_3_month_average
  , cusip
  , description
  , exchange
  , asset_type
  , file_date as market_datetime
from ranked
where rn = 1
