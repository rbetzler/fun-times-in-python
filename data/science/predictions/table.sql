create table dev.predictions (
      model_id                  varchar
    , market_datetime           date
    , symbol                    varchar
    , target                    numeric(20,6)
    , prediction                numeric(20,6)
    , scaled_target             numeric(20,6)
    , scaled_prediction         numeric(20,6)
    , file_datetime             timestamp
    , ingest_datetime           timestamp
);

create table prod.predictions (like dev.predictions);
