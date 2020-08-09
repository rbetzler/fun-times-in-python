from finance.data import sql


class BlackScholesSQLRunner(sql.SQLRunner):
    @property
    def table_name(self) -> str:
        return 'black_scholes'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = '''
            create table td.black_scholes (
                symbol text,
                implied_volatility numeric(20,6),
                strike numeric(20,6),
                days_to_maturity numeric(20,6),
                put_call text,
                file_datetime timestamp without time zone,
                ingest_datetime timestamp without time zone
            )
                partition by range (file_datetime);

            create table if not exists td.black_scholes_20200407 partition of td.black_scholes for values from ('2020-04-07') to ('2020-04-14');
            create table if not exists td.black_scholes_20200414 partition of td.black_scholes for values from ('2020-04-14') to ('2020-04-21');
            create table if not exists td.black_scholes_20200421 partition of td.black_scholes for values from ('2020-04-21') to ('2020-04-28');
            create table if not exists td.black_scholes_20200428 partition of td.black_scholes for values from ('2020-04-28') to ('2020-05-05');
            create table if not exists td.black_scholes_20200505 partition of td.black_scholes for values from ('2020-05-05') to ('2020-05-12');
            create table if not exists td.black_scholes_20200512 partition of td.black_scholes for values from ('2020-05-12') to ('2020-05-19');
            create table if not exists td.black_scholes_20200519 partition of td.black_scholes for values from ('2020-05-19') to ('2020-05-26');
            create table if not exists td.black_scholes_20200526 partition of td.black_scholes for values from ('2020-05-26') to ('2020-06-02');
            create table if not exists td.black_scholes_20200602 partition of td.black_scholes for values from ('2020-06-02') to ('2020-06-09');
            create table if not exists td.black_scholes_20200609 partition of td.black_scholes for values from ('2020-06-09') to ('2020-06-16');
            create table if not exists td.black_scholes_20200616 partition of td.black_scholes for values from ('2020-06-16') to ('2020-06-23');
            create table if not exists td.black_scholes_20200623 partition of td.black_scholes for values from ('2020-06-23') to ('2020-06-30');
            create table if not exists td.black_scholes_20200630 partition of td.black_scholes for values from ('2020-06-30') to ('2020-07-07');
            create table if not exists td.black_scholes_20200707 partition of td.black_scholes for values from ('2020-07-07') to ('2020-07-14');
            create table if not exists td.black_scholes_20200714 partition of td.black_scholes for values from ('2020-07-14') to ('2020-07-21');
            create table if not exists td.black_scholes_20200721 partition of td.black_scholes for values from ('2020-07-21') to ('2020-07-28');
            create table if not exists td.black_scholes_20200728 partition of td.black_scholes for values from ('2020-07-28') to ('2020-08-04');
            create table if not exists td.black_scholes_20200804 partition of td.black_scholes for values from ('2020-08-04') to ('2020-08-11');
            create table if not exists td.black_scholes_20200811 partition of td.black_scholes for values from ('2020-08-11') to ('2020-08-18');
            create table if not exists td.black_scholes_20200818 partition of td.black_scholes for values from ('2020-08-18') to ('2020-08-25');
            create table if not exists td.black_scholes_20200825 partition of td.black_scholes for values from ('2020-08-25') to ('2020-09-01');
            create table if not exists td.black_scholes_20200901 partition of td.black_scholes for values from ('2020-09-01') to ('2020-09-08');
            create table if not exists td.black_scholes_20200908 partition of td.black_scholes for values from ('2020-09-08') to ('2020-09-15');
            create table if not exists td.black_scholes_20200915 partition of td.black_scholes for values from ('2020-09-15') to ('2020-09-22');
            create table if not exists td.black_scholes_20200922 partition of td.black_scholes for values from ('2020-09-22') to ('2020-09-29');
            create table if not exists td.black_scholes_20200929 partition of td.black_scholes for values from ('2020-09-29') to ('2020-10-06');
            create table if not exists td.black_scholes_20201006 partition of td.black_scholes for values from ('2020-10-06') to ('2020-10-13');
            create table if not exists td.black_scholes_20201013 partition of td.black_scholes for values from ('2020-10-13') to ('2020-10-20');
            create table if not exists td.black_scholes_20201020 partition of td.black_scholes for values from ('2020-10-20') to ('2020-10-27');
            create table if not exists td.black_scholes_20201027 partition of td.black_scholes for values from ('2020-10-27') to ('2020-11-03');
            create table if not exists td.black_scholes_20201103 partition of td.black_scholes for values from ('2020-11-03') to ('2020-11-10');
            create table if not exists td.black_scholes_20201110 partition of td.black_scholes for values from ('2020-11-10') to ('2020-11-17');
            create table if not exists td.black_scholes_20201117 partition of td.black_scholes for values from ('2020-11-17') to ('2020-11-24');
            create table if not exists td.black_scholes_20201124 partition of td.black_scholes for values from ('2020-11-24') to ('2020-12-01');
            create table if not exists td.black_scholes_20201201 partition of td.black_scholes for values from ('2020-12-01') to ('2020-12-08');
            create table if not exists td.black_scholes_20201208 partition of td.black_scholes for values from ('2020-12-08') to ('2020-12-15');
            create table if not exists td.black_scholes_20201215 partition of td.black_scholes for values from ('2020-12-15') to ('2020-12-22');
            create table if not exists td.black_scholes_20201222 partition of td.black_scholes for values from ('2020-12-22') to ('2020-12-29');
            create table if not exists td.black_scholes_20201229 partition of td.black_scholes for values from ('2020-12-29') to ('2021-01-05');

            create index if not exists black_scholes_symbol_idx on td.black_scholes (symbol);
            create index if not exists black_scholes_file_datetime_idx on td.black_scholes (file_datetime);
            '''
        return ddl

    @property
    def sql_script(self):
        return None


if __name__ == '__main__':
    BlackScholesSQLRunner().execute()
