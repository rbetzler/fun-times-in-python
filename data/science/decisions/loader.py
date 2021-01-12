import abc
from data.science import loader


class DecisionsLoader(loader.ScienceLoader, abc.ABC):
    @property
    def table(self) -> str:
        return 'decisions'

    @property
    def import_file_prefix(self) -> str:
        return 'd'

    @property
    def columns(self) -> list:
        return [
            'model_id',
            'decisioner_id',
            'model_datetime',
            'market_datetime',
            'symbol',
            'thirty_day_low_prediction',
            'close',
            'put_call',
            'days_to_expiration',
            'strike',
            'price',
            'potential_annual_return',
            'oom_percent',
            'is_sufficiently_profitable',
            'is_sufficiently_oom',
            'is_strike_below_predicted_low_price',
            'quantity',
            'asset',
            'direction',
            'first_order_difference',
            'smoothed_first_order_difference',
            'probability_of_profit',
            'kelly_criterion',
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
