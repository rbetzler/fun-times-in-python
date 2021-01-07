import abc
import pandas as pd
from data import loader


class ScienceLoader(loader.FileIngestion, abc.ABC):
    @property
    @abc.abstractmethod
    def environment(self) -> str:
        pass

    @property
    def job_name(self) -> str:
        return f'{self.environment}_{self.table}_loader'

    @property
    def directory(self) -> str:
        return f'science/{self.environment}/{self.table}'

    @property
    def schema(self) -> str:
        return self.environment

    @property
    @abc.abstractmethod
    def columns(self) -> list:
        pass

    def clean_df(self, df) -> pd.DataFrame:
        df = df[self.columns]
        return df
