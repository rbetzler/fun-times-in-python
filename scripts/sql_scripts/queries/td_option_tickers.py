QUERY = """
    SELECT ticker
    FROM nasdaq.listed_stocks
    WHERE ticker !~ '[\^.~]'
    AND CHARACTER_LENGTH(ticker) BETWEEN 1 AND 4
    AND LEFT(ticker, 1) = 'A'
    LIMIT 3
    """