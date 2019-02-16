select
	ticker,
	date_time,
	open,
	high,
	low,
	close,
	adj_close,
	volume,
	dividend,
	split_numerator,
	split_denominator,
	unix_timestamp
from fact_yahoo_stocks
where ticker = 'AAPL'
order by date_time
limit 8000
