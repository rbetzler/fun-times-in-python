import abc
from finance.data.science import loader


class DecisionsLoader(loader.ScienceLoader, abc.ABC):
    @property
    def table(self) -> str:
        return 'decisions'

    @property
    def import_file_prefix(self) -> str:
        return 'z'

    @property
    def columns(self) -> list:
        return [
            'model_id',
            'model_datetime',
            'direction',
            'asset',
            'target',
            'denormalized_target',
            'prediction',
            'denormalized_prediction',
            'quantity',
            'symbol',
            'file_datetime',
        ]


class DevDecisionsLoader(DecisionsLoader):
    @property
    def environment(self) -> str:
        return 'dev'


class ProdDecisionsLoader(DecisionsLoader):
    @property
    def environment(self) -> str:
        return 'prod'


if __name__ == '__main__':
    DevDecisionsLoader().execute()
    # ProdDecisionsLoader().execute()
