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

create table dev.pred_autoencoder (
    market_datetime date
  , symbol varchar
  , model_id varchar
  , open_1_target numeric(20,6)
  , open_2_target numeric(20,6)
  , open_3_target numeric(20,6)
  , open_4_target numeric(20,6)
  , open_5_target numeric(20,6)
  , open_6_target numeric(20,6)
  , open_7_target numeric(20,6)
  , open_8_target numeric(20,6)
  , open_9_target numeric(20,6)
  , open_10_target numeric(20,6)
  , open_11_target numeric(20,6)
  , open_12_target numeric(20,6)
  , open_13_target numeric(20,6)
  , open_14_target numeric(20,6)
  , open_15_target numeric(20,6)
  , open_16_target numeric(20,6)
  , open_17_target numeric(20,6)
  , open_18_target numeric(20,6)
  , open_19_target numeric(20,6)
  , open_20_target numeric(20,6)
  , open_21_target numeric(20,6)
  , open_22_target numeric(20,6)
  , open_23_target numeric(20,6)
  , open_24_target numeric(20,6)
  , open_25_target numeric(20,6)
  , open_26_target numeric(20,6)
  , open_27_target numeric(20,6)
  , open_28_target numeric(20,6)
  , open_29_target numeric(20,6)
  , open_1_prediction numeric(20,6)
  , open_2_prediction numeric(20,6)
  , open_3_prediction numeric(20,6)
  , open_4_prediction numeric(20,6)
  , open_5_prediction numeric(20,6)
  , open_6_prediction numeric(20,6)
  , open_7_prediction numeric(20,6)
  , open_8_prediction numeric(20,6)
  , open_9_prediction numeric(20,6)
  , open_10_prediction numeric(20,6)
  , open_11_prediction numeric(20,6)
  , open_12_prediction numeric(20,6)
  , open_13_prediction numeric(20,6)
  , open_14_prediction numeric(20,6)
  , open_15_prediction numeric(20,6)
  , open_16_prediction numeric(20,6)
  , open_17_prediction numeric(20,6)
  , open_18_prediction numeric(20,6)
  , open_19_prediction numeric(20,6)
  , open_20_prediction numeric(20,6)
  , open_21_prediction numeric(20,6)
  , open_22_prediction numeric(20,6)
  , open_23_prediction numeric(20,6)
  , open_24_prediction numeric(20,6)
  , open_25_prediction numeric(20,6)
  , open_26_prediction numeric(20,6)
  , open_27_prediction numeric(20,6)
  , open_28_prediction numeric(20,6)
  , open_29_prediction numeric(20,6)
  , file_datetime timestamp
  , ingest_datetime timestamp
);
