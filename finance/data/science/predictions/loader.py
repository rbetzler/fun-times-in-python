import abc
from finance.data.science import loader


class PredictionsLoader(loader.ScienceLoader, abc.ABC):
    @property
    def table(self) -> str:
        return 'predictions'

    @property
    def import_file_prefix(self) -> str:
        return 's'

    @property
    def columns(self) -> list:
        return [
            'model_id',
            'market_datetime',
            'symbol',
            'target',
            'denormalized_target',
            'prediction',
            'denormalized_prediction',
            'normalization_min',
            'normalization_max',
            'file_datetime',
        ]


class DevPredictionsLoader(PredictionsLoader):
    @property
    def environment(self) -> str:
        return 'dev'


class ProdPredictionsLoader(PredictionsLoader):
    @property
    def environment(self) -> str:
        return 'prod'


if __name__ == '__main__':
    DevPredictionsLoader().execute()
    # ProdPredictionsLoader().execute()
