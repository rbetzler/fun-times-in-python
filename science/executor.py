import argparse
import importlib
import inspect

from datetime import datetime
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


def parse(
        args,
        cls_kwargs: set,
) -> dict:
    """Parse cli arguments into kwargs dict"""
    kwargs = {}
    for key, arg in args.__dict__.items():
        if arg is not None and key != 'job' and key in cls_kwargs:
            if key == 'start_date':
                arg = datetime.strptime(arg, '%Y-%m-%d').date()
                assert arg.weekday() < 5, 'Start date must be a weekday'
            kwargs.update({key: arg})
    return kwargs


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
    job_id = args.job
    cls = get_class(job_id=job_id)
    cls_kwargs = get_class_kwargs(cls)
    kwargs = parse(args=args, cls_kwargs=cls_kwargs)

    cls(**kwargs).execute()


if __name__ == '__main__':
    main()
