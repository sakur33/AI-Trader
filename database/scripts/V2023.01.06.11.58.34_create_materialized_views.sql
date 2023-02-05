/* ****************************************************************
 * Author: Alejandro
 * Description: create_materialized_views
 * ****************************************************************
 */
CREATE MATERIALIZED VIEW candle_short(ctmstring, symbol, short_ma) WITH (timescaledb.continuous) AS
SELECT time_bucket('{df[short_ma]}min', ctmstring),
    max(symbol),
    avg(close)
FROM candles
where symbol = 'df[symbol]'
GROUP BY time_bucket('{df[short_ma]}min', ctmstring) WITH NO DATA AUTO REFRESH YES;
CREATE MATERIALIZED VIEW candle_long(ctmstring, symbol, long_ma) WITH (timescaledb.continuous) AS
SELECT time_bucket('{df[long_ma]}min', ctmstring),
    max(symbol),
    avg(close)
FROM candles
where symbol = 'df[symbol]'
GROUP BY time_bucket('{df[long_ma]}min', ctmstring) WITH NO DATA AUTO REFRESH YES;