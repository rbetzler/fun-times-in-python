create table dev.predictions (
      model_id                  varchar
    , market_datetime           date
    , symbol                    varchar
    , target                    numeric(20,6)
    , denormalized_target       numeric(20,6)
    , prediction                numeric(20,6)
    , denormalized_prediction   numeric(20,6)
    , normalization_min         numeric(20,6)
    , normalization_max         numeric(20,6)
    , file_datetime             timestamp
    , ingest_datetime           timestamp
);

create table prod.predictions (like dev.predictions);
