import argparse
import importlib
import inspect

import pandas as pd
from datetime import datetime, timedelta
from science import core
from science.utilities import modeling_utils


def get_class(job_id: str) -> classmethod:
    """Determine which class to run"""
    if job_id[0] == 's':
        job_type = 'predictor'
    elif job_id[0] == 'd':
        job_type = 'decisioner'
    else:
        raise NotImplementedError('Missing job_id arg')

    module = importlib.import_module(f'science.{job_type.lower()}.{job_id}')
    cls = getattr(module, f'{job_id.upper()}')
    return cls


# TODO: Find a more elegant solution to cls inspection
def get_class_kwargs(cls: classmethod) -> set:
    """Get keyword arguments for classes"""
    sub_cls_kwargs = inspect.getfullargspec(cls.__init__).args
    base_cls_kwargs = inspect.getfullargspec(core.Science.__init__).args
    cls_kwargs = set(base_cls_kwargs + sub_cls_kwargs)
    return cls_kwargs


def configure_backtest(run_kwargs: dict) -> list:
    start_date = run_kwargs['start_date']
    end_date = start_date + timedelta(days=int(run_kwargs['n_days']))
    run_kwargs.update({'n_days': 0})

    days = pd.date_range(
        start=start_date,
        end=end_date,
        freq='b',
    ).to_list()

    kwargs = []
    for day in days:
        run_kwargs.update({'start_date': day.to_pydatetime().date()})
        kwargs.append(run_kwargs.copy())
    return kwargs


def parse(
        args,
        cls_kwargs: set,
) -> list:
    """Parse cli arguments into a list of kwargs dicts"""
    run_kwargs = {}
    for key, arg in args.__dict__.items():
        if arg is not None and key != 'job' and key in cls_kwargs:
            if key == 'start_date':
                arg = datetime.strptime(arg, '%Y-%m-%d').date()
                # TODO: Assert on stock market holidays as well
                assert arg.weekday() < 5, 'Start date must be a weekday'
            run_kwargs.update({key: arg})

    if args.is_backtest:
        kwargs = configure_backtest(run_kwargs)
    else:
        kwargs = [run_kwargs]
    return kwargs


def main():
    """Argument parser from science jobs"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-j',
        '--job',
        help='Which job to run: stock, decisioner.',
    )

    parser.add_argument(
        '-s',
        '--start_date',
        default=modeling_utils.get_latest_market_datetime(),
        help='First date for training or testing. Must be a weekday',
    )

    parser.add_argument(
        '-n',
        '--n_days',
        default=0,
        help='How many days to include, whether for training or backtesting.',
    )

    parser.add_argument(
        '-b',
        '--is_backtest',
        action='store_true',
        help='Whether to run a backtest.',
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
        help='Whether to save output files.',
    )

    parser.add_argument(
        '-t',
        '--is_training_run',
        action='store_true',
        help='Whether to train the lstm.',
    )

    args = parser.parse_args()
    job_id = args.job
    cls = get_class(job_id=job_id)
    cls_kwargs = get_class_kwargs(cls)
    kwargs = parse(args=args, cls_kwargs=cls_kwargs)
    for k in kwargs:
        cls(**k).execute()


if __name__ == '__main__':
    main()
