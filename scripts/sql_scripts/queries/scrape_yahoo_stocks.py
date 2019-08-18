QUERY = """
  select
    ticker,
    '34450' as start_date,
    '153145400400' as end_date
  from dw_stocks.dim_stocks 
  where ticker !~ '[\^.~]'
    and character_length(ticker) between 1 and 4
  union
  select
    ticker,
    max(trunc(unix_timestamp::numeric, 0))::varchar as start_date,
    '153145400400' as end_date
  from dw_stocks.fact_yahoo_stocks
  group by ticker
  """
