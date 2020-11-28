import argparse
import inspect

from datetime import datetime
from finance.science.dev import decisioner, stock_predictor, volatility_predictor
from finance.science.utilities import modeling_utils


def main():
    """Argument parser from science jobs"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-j',
        '--job',
        help='Which job to run: stock, volatility, decisioner.',
    )

    parser.add_argument(
        '-s',
        '--start_date',
        default=modeling_utils.get_latest_market_datetime(),
        help='First date for training or testing.',
    )

    parser.add_argument(
        '-n',
        '--n_days',
        default=0,
        help='How many days to include.',
    )

    parser.add_argument(
        '-p',
        '--is_prod',
        action='store_true',
        help='Whether to save the files in prod.',
    )

    parser.add_argument(
        '-a',
        '--archive_files',
        action='store_true',
        help='Whether to save prediction and trade files.',
    )

    parser.add_argument(
        '-t',
        '--is_training_run',
        action='store_true',
        help='Whether to train the lstm.',
    )

    # Determine which job to run
    args = parser.parse_args()
    job_id = args.job[0]
    if job_id == 's':
        cls = stock_predictor.StockPredictor
    elif job_id == 'v':
        cls = volatility_predictor.VolatilityPredictor
    elif job_id == 'd':
        cls = decisioner.StockDecisioner
    cls_kwargs = inspect.getfullargspec(cls.__init__).args

    # Parse args
    kwargs = {}
    for key, arg in args.__dict__.items():
        if arg is not None and key != 'job' and key in cls_kwargs:
            if key == 'start_date':
                arg = datetime.strptime(arg, '%Y-%m-%d').date()
            kwargs.update({key: arg})

    cls(**kwargs).execute()


if __name__ == '__main__':
    main()
