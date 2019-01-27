select *
from fact_yahoo_stocks
where ticker in
	(select ticker
	from fact_yahoo_stocks
	group by 1
	having count(*) > 500
	limit 1)
