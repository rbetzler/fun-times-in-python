QUERY = """
    SELECT DISTINCT ticker
    FROM nasdaq.listed_stocks
    WHERE ticker !~ '[\^.~]'
    AND CHARACTER_LENGTH(ticker) BETWEEN 1 AND 4
    AND ticker IN ('GIS', 'GE', 'KO')
    LIMIT 3
    """