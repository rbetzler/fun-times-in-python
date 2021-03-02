import abc
from data.science import loader


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
            'prediction',
            'scaled_target',
            'scaled_prediction',
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


class DevPredAutoencoderLoader(loader.ScienceLoader):
    @property
    def table(self) -> str:
        return 'pred_autoencoder'

    @property
    def import_file_prefix(self) -> str:
        return 'a'

    @property
    def environment(self) -> str:
        return 'dev'

    @property
    def columns(self) -> list:
        return [
            'market_datetime',
            'symbol',
            'model_id',
            'open_1_target',
            'open_2_target',
            'open_3_target',
            'open_4_target',
            'open_5_target',
            'open_6_target',
            'open_7_target',
            'open_8_target',
            'open_9_target',
            'open_10_target',
            'open_11_target',
            'open_12_target',
            'open_13_target',
            'open_14_target',
            'open_15_target',
            'open_16_target',
            'open_17_target',
            'open_18_target',
            'open_19_target',
            'open_20_target',
            'open_21_target',
            'open_22_target',
            'open_23_target',
            'open_24_target',
            'open_25_target',
            'open_26_target',
            'open_27_target',
            'open_28_target',
            'open_29_target',
            'open_1_prediction',
            'open_2_prediction',
            'open_3_prediction',
            'open_4_prediction',
            'open_5_prediction',
            'open_6_prediction',
            'open_7_prediction',
            'open_8_prediction',
            'open_9_prediction',
            'open_10_prediction',
            'open_11_prediction',
            'open_12_prediction',
            'open_13_prediction',
            'open_14_prediction',
            'open_15_prediction',
            'open_16_prediction',
            'open_17_prediction',
            'open_18_prediction',
            'open_19_prediction',
            'open_20_prediction',
            'open_21_prediction',
            'open_22_prediction',
            'open_23_prediction',
            'open_24_prediction',
            'open_25_prediction',
            'open_26_prediction',
            'open_27_prediction',
            'open_28_prediction',
            'open_29_prediction',
            'file_datetime',
        ]


if __name__ == '__main__':
    DevPredictionsLoader().execute()
    # DevPredAutoencoderLoader().execute()
    # ProdPredictionsLoader().execute()
