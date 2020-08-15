import argparse
import engine


def main():
    """Argument parser from lstm engine"""

    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    kwargs = {
            'is_prod': args.is_prod,
            'archive_files': args.archive_files,
            'is_training_run': args.is_training_run,
        }

    engine.Dev(**kwargs).execute()


if __name__ == '__main__':
    main()
