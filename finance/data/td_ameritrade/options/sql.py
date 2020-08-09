from finance.data import sql


class TdOptionsSQLRunner(sql.SQLRunner):
    @property
    def table_name(self) -> str:
        return 'options'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = '''
            create table td.options (
                symbol                      text,
                volatility                  numeric(20,6),
                n_contracts                 numeric(20,6),
                interest_rate               numeric(20,6),
                put_call                    text,
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
                delta                       numeric(60,20),
                gamma                       numeric(60,20),
                theta                       numeric(60,20),
                vega                        numeric(60,20),
                rho                         numeric(60,20),
                open_interest               numeric(20,6),
                time_value                  numeric(20,6),
                theoretical_option_value    numeric(20,6),
                theoretical_volatility      numeric(20,6),
                strike_price                numeric(20,6),
                expiration_date             text,
                days_to_expiration          numeric(20,6),
                expiration_date_from_epoch  timestamp without time zone,
                strike                      numeric(20,6),
                strike_date                 timestamp without time zone,
                days_to_expiration_date     numeric(20,6),
                file_datetime               timestamp without time zone,
                ingest_datetime             timestamp without time zone
            );

            create index if not exists options_symbol_idx on td.options (symbol);
            create index if not exists options_file_datetime_idx on td.options (file_datetime);

            create table td.options_raw (
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
            )
                partition by range (file_datetime);

            create table if not exists td.options_raw_20000101 partition of td.options_raw for values from ('2000-01-01') to ('2019-01-01');
            create table if not exists td.options_raw_20190101 partition of td.options_raw for values from ('2019-01-01') to ('2019-01-08');
            create table if not exists td.options_raw_20190108 partition of td.options_raw for values from ('2019-01-08') to ('2019-01-15');
            create table if not exists td.options_raw_20190115 partition of td.options_raw for values from ('2019-01-15') to ('2019-01-22');
            create table if not exists td.options_raw_20190122 partition of td.options_raw for values from ('2019-01-22') to ('2019-01-29');
            create table if not exists td.options_raw_20190129 partition of td.options_raw for values from ('2019-01-29') to ('2019-02-05');
            create table if not exists td.options_raw_20190205 partition of td.options_raw for values from ('2019-02-05') to ('2019-02-12');
            create table if not exists td.options_raw_20190212 partition of td.options_raw for values from ('2019-02-12') to ('2019-02-19');
            create table if not exists td.options_raw_20190219 partition of td.options_raw for values from ('2019-02-19') to ('2019-02-26');
            create table if not exists td.options_raw_20190226 partition of td.options_raw for values from ('2019-02-26') to ('2019-03-05');
            create table if not exists td.options_raw_20190305 partition of td.options_raw for values from ('2019-03-05') to ('2019-03-12');
            create table if not exists td.options_raw_20190312 partition of td.options_raw for values from ('2019-03-12') to ('2019-03-19');
            create table if not exists td.options_raw_20190319 partition of td.options_raw for values from ('2019-03-19') to ('2019-03-26');
            create table if not exists td.options_raw_20190326 partition of td.options_raw for values from ('2019-03-26') to ('2019-04-02');
            create table if not exists td.options_raw_20190402 partition of td.options_raw for values from ('2019-04-02') to ('2019-04-09');
            create table if not exists td.options_raw_20190409 partition of td.options_raw for values from ('2019-04-09') to ('2019-04-16');
            create table if not exists td.options_raw_20190416 partition of td.options_raw for values from ('2019-04-16') to ('2019-04-23');
            create table if not exists td.options_raw_20190423 partition of td.options_raw for values from ('2019-04-23') to ('2019-04-30');
            create table if not exists td.options_raw_20190430 partition of td.options_raw for values from ('2019-04-30') to ('2019-05-07');
            create table if not exists td.options_raw_20190507 partition of td.options_raw for values from ('2019-05-07') to ('2019-05-14');
            create table if not exists td.options_raw_20190514 partition of td.options_raw for values from ('2019-05-14') to ('2019-05-21');
            create table if not exists td.options_raw_20190521 partition of td.options_raw for values from ('2019-05-21') to ('2019-05-28');
            create table if not exists td.options_raw_20190528 partition of td.options_raw for values from ('2019-05-28') to ('2019-06-04');
            create table if not exists td.options_raw_20190604 partition of td.options_raw for values from ('2019-06-04') to ('2019-06-11');
            create table if not exists td.options_raw_20190611 partition of td.options_raw for values from ('2019-06-11') to ('2019-06-18');
            create table if not exists td.options_raw_20190618 partition of td.options_raw for values from ('2019-06-18') to ('2019-06-25');
            create table if not exists td.options_raw_20190625 partition of td.options_raw for values from ('2019-06-25') to ('2019-07-02');
            create table if not exists td.options_raw_20190702 partition of td.options_raw for values from ('2019-07-02') to ('2019-07-09');
            create table if not exists td.options_raw_20190709 partition of td.options_raw for values from ('2019-07-09') to ('2019-07-16');
            create table if not exists td.options_raw_20190716 partition of td.options_raw for values from ('2019-07-16') to ('2019-07-23');
            create table if not exists td.options_raw_20190723 partition of td.options_raw for values from ('2019-07-23') to ('2019-07-30');
            create table if not exists td.options_raw_20190730 partition of td.options_raw for values from ('2019-07-30') to ('2019-08-06');
            create table if not exists td.options_raw_20190806 partition of td.options_raw for values from ('2019-08-06') to ('2019-08-13');
            create table if not exists td.options_raw_20190813 partition of td.options_raw for values from ('2019-08-13') to ('2019-08-20');
            create table if not exists td.options_raw_20190820 partition of td.options_raw for values from ('2019-08-20') to ('2019-08-27');
            create table if not exists td.options_raw_20190827 partition of td.options_raw for values from ('2019-08-27') to ('2019-09-03');
            create table if not exists td.options_raw_20190903 partition of td.options_raw for values from ('2019-09-03') to ('2019-09-10');
            create table if not exists td.options_raw_20190910 partition of td.options_raw for values from ('2019-09-10') to ('2019-09-17');
            create table if not exists td.options_raw_20190917 partition of td.options_raw for values from ('2019-09-17') to ('2019-09-24');
            create table if not exists td.options_raw_20190924 partition of td.options_raw for values from ('2019-09-24') to ('2019-10-01');
            create table if not exists td.options_raw_20191001 partition of td.options_raw for values from ('2019-10-01') to ('2019-10-08');
            create table if not exists td.options_raw_20191008 partition of td.options_raw for values from ('2019-10-08') to ('2019-10-15');
            create table if not exists td.options_raw_20191015 partition of td.options_raw for values from ('2019-10-15') to ('2019-10-22');
            create table if not exists td.options_raw_20191022 partition of td.options_raw for values from ('2019-10-22') to ('2019-10-29');
            create table if not exists td.options_raw_20191029 partition of td.options_raw for values from ('2019-10-29') to ('2019-11-05');
            create table if not exists td.options_raw_20191105 partition of td.options_raw for values from ('2019-11-05') to ('2019-11-12');
            create table if not exists td.options_raw_20191112 partition of td.options_raw for values from ('2019-11-12') to ('2019-11-19');
            create table if not exists td.options_raw_20191119 partition of td.options_raw for values from ('2019-11-19') to ('2019-11-26');
            create table if not exists td.options_raw_20191126 partition of td.options_raw for values from ('2019-11-26') to ('2019-12-03');
            create table if not exists td.options_raw_20191203 partition of td.options_raw for values from ('2019-12-03') to ('2019-12-10');
            create table if not exists td.options_raw_20191210 partition of td.options_raw for values from ('2019-12-10') to ('2019-12-17');
            create table if not exists td.options_raw_20191217 partition of td.options_raw for values from ('2019-12-17') to ('2019-12-24');
            create table if not exists td.options_raw_20191224 partition of td.options_raw for values from ('2019-12-24') to ('2019-12-31');
            create table if not exists td.options_raw_20191231 partition of td.options_raw for values from ('2019-12-31') to ('2020-01-07');
            create table if not exists td.options_raw_20200107 partition of td.options_raw for values from ('2020-01-07') to ('2020-01-14');
            create table if not exists td.options_raw_20200114 partition of td.options_raw for values from ('2020-01-14') to ('2020-01-21');
            create table if not exists td.options_raw_20200121 partition of td.options_raw for values from ('2020-01-21') to ('2020-01-28');
            create table if not exists td.options_raw_20200128 partition of td.options_raw for values from ('2020-01-28') to ('2020-02-04');
            create table if not exists td.options_raw_20200204 partition of td.options_raw for values from ('2020-02-04') to ('2020-02-11');
            create table if not exists td.options_raw_20200211 partition of td.options_raw for values from ('2020-02-11') to ('2020-02-18');
            create table if not exists td.options_raw_20200218 partition of td.options_raw for values from ('2020-02-18') to ('2020-02-25');
            create table if not exists td.options_raw_20200225 partition of td.options_raw for values from ('2020-02-25') to ('2020-03-03');
            create table if not exists td.options_raw_20200303 partition of td.options_raw for values from ('2020-03-03') to ('2020-03-10');
            create table if not exists td.options_raw_20200310 partition of td.options_raw for values from ('2020-03-10') to ('2020-03-17');
            create table if not exists td.options_raw_20200317 partition of td.options_raw for values from ('2020-03-17') to ('2020-03-24');
            create table if not exists td.options_raw_20200324 partition of td.options_raw for values from ('2020-03-24') to ('2020-03-31');
            create table if not exists td.options_raw_20200331 partition of td.options_raw for values from ('2020-03-31') to ('2020-04-07');
            create table if not exists td.options_raw_20200407 partition of td.options_raw for values from ('2020-04-07') to ('2020-04-14');
            create table if not exists td.options_raw_20200414 partition of td.options_raw for values from ('2020-04-14') to ('2020-04-21');
            create table if not exists td.options_raw_20200421 partition of td.options_raw for values from ('2020-04-21') to ('2020-04-28');
            create table if not exists td.options_raw_20200428 partition of td.options_raw for values from ('2020-04-28') to ('2020-05-05');
            create table if not exists td.options_raw_20200505 partition of td.options_raw for values from ('2020-05-05') to ('2020-05-12');
            create table if not exists td.options_raw_20200512 partition of td.options_raw for values from ('2020-05-12') to ('2020-05-19');
            create table if not exists td.options_raw_20200519 partition of td.options_raw for values from ('2020-05-19') to ('2020-05-26');
            create table if not exists td.options_raw_20200526 partition of td.options_raw for values from ('2020-05-26') to ('2020-06-02');
            create table if not exists td.options_raw_20200602 partition of td.options_raw for values from ('2020-06-02') to ('2020-06-09');
            create table if not exists td.options_raw_20200609 partition of td.options_raw for values from ('2020-06-09') to ('2020-06-16');
            create table if not exists td.options_raw_20200616 partition of td.options_raw for values from ('2020-06-16') to ('2020-06-23');
            create table if not exists td.options_raw_20200623 partition of td.options_raw for values from ('2020-06-23') to ('2020-06-30');
            create table if not exists td.options_raw_20200630 partition of td.options_raw for values from ('2020-06-30') to ('2020-07-07');
            create table if not exists td.options_raw_20200707 partition of td.options_raw for values from ('2020-07-07') to ('2020-07-14');
            create table if not exists td.options_raw_20200714 partition of td.options_raw for values from ('2020-07-14') to ('2020-07-21');
            create table if not exists td.options_raw_20200721 partition of td.options_raw for values from ('2020-07-21') to ('2020-07-28');
            create table if not exists td.options_raw_20200728 partition of td.options_raw for values from ('2020-07-28') to ('2020-08-04');
            create table if not exists td.options_raw_20200804 partition of td.options_raw for values from ('2020-08-04') to ('2020-08-11');
            create table if not exists td.options_raw_20200811 partition of td.options_raw for values from ('2020-08-11') to ('2020-08-18');
            create table if not exists td.options_raw_20200818 partition of td.options_raw for values from ('2020-08-18') to ('2020-08-25');
            create table if not exists td.options_raw_20200825 partition of td.options_raw for values from ('2020-08-25') to ('2020-09-01');
            create table if not exists td.options_raw_20200901 partition of td.options_raw for values from ('2020-09-01') to ('2020-09-08');
            create table if not exists td.options_raw_20200908 partition of td.options_raw for values from ('2020-09-08') to ('2020-09-15');
            create table if not exists td.options_raw_20200915 partition of td.options_raw for values from ('2020-09-15') to ('2020-09-22');
            create table if not exists td.options_raw_20200922 partition of td.options_raw for values from ('2020-09-22') to ('2020-09-29');
            create table if not exists td.options_raw_20200929 partition of td.options_raw for values from ('2020-09-29') to ('2020-10-06');
            create table if not exists td.options_raw_20201006 partition of td.options_raw for values from ('2020-10-06') to ('2020-10-13');
            create table if not exists td.options_raw_20201013 partition of td.options_raw for values from ('2020-10-13') to ('2020-10-20');
            create table if not exists td.options_raw_20201020 partition of td.options_raw for values from ('2020-10-20') to ('2020-10-27');
            create table if not exists td.options_raw_20201027 partition of td.options_raw for values from ('2020-10-27') to ('2020-11-03');
            create table if not exists td.options_raw_20201103 partition of td.options_raw for values from ('2020-11-03') to ('2020-11-10');
            create table if not exists td.options_raw_20201110 partition of td.options_raw for values from ('2020-11-10') to ('2020-11-17');
            create table if not exists td.options_raw_20201117 partition of td.options_raw for values from ('2020-11-17') to ('2020-11-24');
            create table if not exists td.options_raw_20201124 partition of td.options_raw for values from ('2020-11-24') to ('2020-12-01');
            create table if not exists td.options_raw_20201201 partition of td.options_raw for values from ('2020-12-01') to ('2020-12-08');
            create table if not exists td.options_raw_20201208 partition of td.options_raw for values from ('2020-12-08') to ('2020-12-15');
            create table if not exists td.options_raw_20201215 partition of td.options_raw for values from ('2020-12-15') to ('2020-12-22');
            create table if not exists td.options_raw_20201222 partition of td.options_raw for values from ('2020-12-22') to ('2020-12-29');
            create table if not exists td.options_raw_20201229 partition of td.options_raw for values from ('2020-12-29') to ('2021-01-05');
            create table if not exists td.options_raw_20210105 partition of td.options_raw for values from ('2021-01-05') to ('2021-01-12');
            create table if not exists td.options_raw_20210112 partition of td.options_raw for values from ('2021-01-12') to ('2021-01-19');
            create table if not exists td.options_raw_20210119 partition of td.options_raw for values from ('2021-01-19') to ('2021-01-26');
            create table if not exists td.options_raw_20210126 partition of td.options_raw for values from ('2021-01-26') to ('2021-02-02');
            create table if not exists td.options_raw_20210202 partition of td.options_raw for values from ('2021-02-02') to ('2021-02-09');
            create table if not exists td.options_raw_20210209 partition of td.options_raw for values from ('2021-02-09') to ('2021-02-16');
            create table if not exists td.options_raw_20210216 partition of td.options_raw for values from ('2021-02-16') to ('2021-02-23');
            create table if not exists td.options_raw_20210223 partition of td.options_raw for values from ('2021-02-23') to ('2021-03-02');
            create table if not exists td.options_raw_20210302 partition of td.options_raw for values from ('2021-03-02') to ('2021-03-09');
            create table if not exists td.options_raw_20210309 partition of td.options_raw for values from ('2021-03-09') to ('2021-03-16');
            create table if not exists td.options_raw_20210316 partition of td.options_raw for values from ('2021-03-16') to ('2021-03-23');
            create table if not exists td.options_raw_20210323 partition of td.options_raw for values from ('2021-03-23') to ('2021-03-30');
            create table if not exists td.options_raw_20210330 partition of td.options_raw for values from ('2021-03-30') to ('2021-04-06');
            create table if not exists td.options_raw_20210406 partition of td.options_raw for values from ('2021-04-06') to ('2021-04-13');
            create table if not exists td.options_raw_20210413 partition of td.options_raw for values from ('2021-04-13') to ('2021-04-20');
            create table if not exists td.options_raw_20210420 partition of td.options_raw for values from ('2021-04-20') to ('2021-04-27');
            create table if not exists td.options_raw_20210427 partition of td.options_raw for values from ('2021-04-27') to ('2021-05-04');
            create table if not exists td.options_raw_20210504 partition of td.options_raw for values from ('2021-05-04') to ('2021-05-11');
            create table if not exists td.options_raw_20210511 partition of td.options_raw for values from ('2021-05-11') to ('2021-05-18');
            create table if not exists td.options_raw_20210518 partition of td.options_raw for values from ('2021-05-18') to ('2021-05-25');
            create table if not exists td.options_raw_20210525 partition of td.options_raw for values from ('2021-05-25') to ('2021-06-01');
            create table if not exists td.options_raw_20210601 partition of td.options_raw for values from ('2021-06-01') to ('2021-06-08');
            create table if not exists td.options_raw_20210608 partition of td.options_raw for values from ('2021-06-08') to ('2021-06-15');
            create table if not exists td.options_raw_20210615 partition of td.options_raw for values from ('2021-06-15') to ('2021-06-22');
            create table if not exists td.options_raw_20210622 partition of td.options_raw for values from ('2021-06-22') to ('2021-06-29');
            create table if not exists td.options_raw_20210629 partition of td.options_raw for values from ('2021-06-29') to ('2021-07-06');
            create table if not exists td.options_raw_20210706 partition of td.options_raw for values from ('2021-07-06') to ('2021-07-13');
            create table if not exists td.options_raw_20210713 partition of td.options_raw for values from ('2021-07-13') to ('2021-07-20');
            create table if not exists td.options_raw_20210720 partition of td.options_raw for values from ('2021-07-20') to ('2021-07-27');
            create table if not exists td.options_raw_20210727 partition of td.options_raw for values from ('2021-07-27') to ('2021-08-03');
            create table if not exists td.options_raw_20210803 partition of td.options_raw for values from ('2021-08-03') to ('2021-08-10');
            create table if not exists td.options_raw_20210810 partition of td.options_raw for values from ('2021-08-10') to ('2021-08-17');
            create table if not exists td.options_raw_20210817 partition of td.options_raw for values from ('2021-08-17') to ('2021-08-24');
            create table if not exists td.options_raw_20210824 partition of td.options_raw for values from ('2021-08-24') to ('2021-08-31');
            create table if not exists td.options_raw_20210831 partition of td.options_raw for values from ('2021-08-31') to ('2021-09-07');
            create table if not exists td.options_raw_20210907 partition of td.options_raw for values from ('2021-09-07') to ('2021-09-14');
            create table if not exists td.options_raw_20210914 partition of td.options_raw for values from ('2021-09-14') to ('2021-09-21');
            create table if not exists td.options_raw_20210921 partition of td.options_raw for values from ('2021-09-21') to ('2021-09-28');
            create table if not exists td.options_raw_20210928 partition of td.options_raw for values from ('2021-09-28') to ('2021-10-05');
            create table if not exists td.options_raw_20211005 partition of td.options_raw for values from ('2021-10-05') to ('2021-10-12');
            create table if not exists td.options_raw_20211012 partition of td.options_raw for values from ('2021-10-12') to ('2021-10-19');
            create table if not exists td.options_raw_20211019 partition of td.options_raw for values from ('2021-10-19') to ('2021-10-26');
            create table if not exists td.options_raw_20211026 partition of td.options_raw for values from ('2021-10-26') to ('2021-11-02');
            create table if not exists td.options_raw_20211102 partition of td.options_raw for values from ('2021-11-02') to ('2021-11-09');
            create table if not exists td.options_raw_20211109 partition of td.options_raw for values from ('2021-11-09') to ('2021-11-16');

            create index options_raw_symbol_idx on td.options_raw (symbol);
            create index options_raw_file_idx on td.options_raw (file_datetime);
            create index options_raw_ingest_idx on td.options_raw (ingest_datetime);
            create index options_raw_symbol_file_ingest_idx on td.options_raw (symbol, date(file_datetime), ingest_datetime desc);
            '''
        return ddl

    @property
    def sql_script(self) -> str:
        script = '''
            truncate td.options;
            insert into td.options (
                with
                dated as (
                    select
                          symbol
                        , volatility
                        , n_contracts
                        , interest_rate
                        , put_call
                        , bid
                        , ask
                        , last
                        , mark
                        , bid_size
                        , ask_size
                        , bid_ask_size
                        , last_size
                        , high_price
                        , low_price
                        , open_price
                        , close_price
                        , total_volume
                        , trade_date
                        , delta
                        , gamma
                        , theta
                        , vega
                        , rho
                        , open_interest
                        , time_value
                        , theoretical_option_value
                        , theoretical_volatility
                        , strike_price
                        , expiration_date
                        , days_to_expiration
                        , expiration_date_from_epoch
                        , strike
                        , strike_date
                        , days_to_expiration_date
                        , file_datetime
                        , ingest_datetime
                        , case
                            -- If Sunday, set to Friday
                            when extract(isodow from file_datetime) = 1 then file_datetime::date - 2
                            -- If Saturday, set to Friday
                            when extract(isodow from file_datetime) = 7 then file_datetime::date - 1
                            -- If after market close, leave as is
                            when extract('hour' from file_datetime) >= 20 then file_datetime::date
                            -- If before market open, set to prior day
                            when extract(hour from file_datetime) < 13 or (extract('hour' from file_datetime) = 13 and extract('minute' from file_datetime) <= 30) then file_datetime::date - 1
                            end as file_date
                    from td.options_raw
                    where file_datetime > current_date - 10 and file_datetime < current_date + 1
                    )
                , ranked as (
                    select *
                        , row_number() over(partition by file_date, symbol, put_call, strike, days_to_expiration order by ingest_datetime desc) as rn
                    from dated
                    where file_date is not null
                    )
                select
                      symbol
                    , volatility
                    , n_contracts
                    , interest_rate
                    , put_call
                    , bid
                    , ask
                    , last
                    , mark
                    , bid_size
                    , ask_size
                    , bid_ask_size
                    , last_size
                    , high_price
                    , low_price
                    , open_price
                    , close_price
                    , total_volume
                    , trade_date
                    , delta
                    , gamma
                    , theta
                    , vega
                    , rho
                    , open_interest
                    , time_value
                    , theoretical_option_value
                    , theoretical_volatility
                    , strike_price
                    , expiration_date
                    , days_to_expiration
                    , expiration_date_from_epoch
                    , strike
                    , strike_date
                    , days_to_expiration_date
                    , file_date as file_datetime
                    , ingest_datetime
                from ranked
                where rn = 1
                );
            '''
        return script


if __name__ == '__main__':
    TdOptionsSQLRunner().execute()
