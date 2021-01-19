create table dev.decisions (
      model_id                             varchar
    , decisioner_id                        varchar
    , model_datetime                       timestamp
    , market_datetime                      timestamp
    , symbol                               varchar
    , thirty_day_low_prediction            numeric(20,6)
    , close                                numeric(20,6)
    , put_call                             varchar
    , days_to_expiration                   numeric(20,6)
    , strike                               numeric(20,6)
    , price                                numeric(20,6)
    , potential_annual_return              numeric(20,6)
    , oom_percent                          numeric(20,6)
    , quantity                             numeric(20,6)
    , asset                                varchar
    , direction                            varchar
    , first_order_difference               numeric(20,6)
    , smoothed_first_order_difference      numeric(20,6)
    , probability_of_profit                numeric(20,6)
    , kelly_criterion                      numeric(20,6)
    , file_datetime                        timestamp
    , ingest_datetime                      timestamp
);

create table prod.decisions (like dev.decisions);
