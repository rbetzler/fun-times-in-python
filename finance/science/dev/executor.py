import argparse
from finance.science.dev import predictor


def main():
    """Argument parser from lstm engine"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--start_date',
        help='Whether to save the files in prod.',
    )

    parser.add_argument(
        '--n_days',
        help='Whether to save the files in prod.',
    )

    parser.add_argument(
        '--is_prod',
        default=False,
        help='Whether to save the files in prod.',
    )

    parser.add_argument(
        '--archive_files',
        default=False,
        help='Whether to save prediction and trade files',
    )

    parser.add_argument(
        '--is_training_run',
        default=False,
        help='Whether to train the lstm',
    )

    # Only pass not none args
    kwargs = {}
    args = parser.parse_args()
    for key, arg in args.__dict__.items():
        if arg is not None:
            kwargs.update({key: arg})

    predictor.Dev(**kwargs).execute()


if __name__ == '__main__':
    main()
