## Science
* Herein are two frameworks:
  * Predictor, which runs an LSTM to generate predictions
  * Decisioner, which selects trades to make
* Both frameworks write files for ingestion into Postgres.

## Running executor.py locally
* Training Run:
  ```
  python science/executor.py --job=s1 --start_date='2015-01-15' --n_days=500 --is_training_run
  ```
* Production Run:
  ```
  python science/executor.py --job=s1
  ```

## Development Process
1. Using fake data, develop models that beat benchmarks, i.e., simple strategies
2. Repeat back test on historical data
3. Shadow deploy and monitor model
4. Fully deploy model
