import abc
import datetime
import os
import pandas as pd
import torch

from science import core
from utilities import modeling_utils, nn_utils, utils


class Predictor(core.Science, abc.ABC):
    def __init__(
            self,
            n_days: int = 1000,
            is_training_run: bool = False,
            n_subrun: int = None,
            device: int = None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.end_date = self.start_date + datetime.timedelta(days=int(n_days))
        self._is_training_run = is_training_run
        self.n_subrun = n_subrun
        self.device = f'cuda:{device}' if device in [0, 1] else 'cuda'

    @property
    def is_training_run(self) -> bool:
        """Whether the model will be trained or not"""
        return self._is_training_run

    @property
    def n_subruns(self) -> int:
        """When backtesting, the number of iterations for a given date range"""
        return 2

    @property
    def limit(self) -> int:
        """When backtesting, the size of the dataset"""
        return 80000

    @property
    def trained_model_filepath(self) -> str:
        """Filepath from where to load and to where to save a trained model"""
        return f'/usr/src/app/audit/science/{self.location}/models/{self.model_id}'

    @property
    def output_subfolder(self) -> str:
        """Filepath from where to load and to where to save a trained model"""
        return 'predictions'

    @abc.abstractmethod
    def model(self):
        """Pytorch nn to implement"""
        pass

    @property
    def hidden_shape(self) -> int:
        """Size of hidden shape"""
        return 100

    @property
    def batch_size(self) -> int:
        """Number of rows per batch"""
        return 40000

    @property
    def sequence_length(self) -> int:
        """Number of sequences"""
        return 2

    @property
    def n_epochs(self) -> int:
        """Number of loops when model fitting"""
        return 500

    @property
    def loss_function(self):
        """Loss function for model fitting"""
        return torch.nn.L1Loss(reduction='sum')

    @property
    def learning_rate(self) -> float:
        """Learning rate for optimizer"""
        return .0001

    def optimizer(self, model):
        """Optimization method to use in model fitting"""
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    @property
    def target_column(self) -> str:
        """Column for model to predict"""
        return 'target'

    @property
    @abc.abstractmethod
    def columns_to_ignore(self) -> list:
        """Columns to exclude from model inputs"""
        pass

    @abc.abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data pre-model run"""
        pass

    @abc.abstractmethod
    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process data post-model run"""
        pass

    def execute(self):
        print(f'''
        Timestamp: {datetime.datetime.utcnow()}
        Model ID: {self.model_id}
        Location: {self.location}
        Training: {self.is_training_run}
        Archive: {self.archive_files}
        Start Date: {self.start_date}
        End Date: {self.end_date}
        Subrun: {self.n_subrun}
        ''')

        print(f'Getting raw data {datetime.datetime.utcnow()}')
        df = utils.query_db(query=self.query)

        # TODO: Raise an error if data is missing; accommodate holidays
        if not df.empty:

            print(f'Pre-processing raw data {datetime.datetime.utcnow()}')
            input = self.preprocess_data(df)
            x = input.drop(self.columns_to_ignore, axis=1)
            y = input[self.target_column]

            print(f'Padding data {datetime.datetime.utcnow()}')
            x_pad, y_pad = nn_utils.pad_data(self.batch_size, x, y)

            print(f'Determining data shape {datetime.datetime.utcnow()}')
            n_batches, input_shape, output_shape = nn_utils.determine_shapes(
                x=x_pad,
                y=y_pad,
                batch_size=self.batch_size,
                sequence_length=self.sequence_length,
            )
            i_shape = input_shape[2]
            o_shape = output_shape[1]

            print(f'''
            Configuring model {datetime.datetime.utcnow()}
                Input shape: {i_shape}
                Hidden shape: {self.hidden_shape}
                Output shape: {o_shape}
            ''')
            model = self.model(
                input_shape=i_shape,
                hidden_shape=self.hidden_shape,
                output_shape=o_shape,
            ).to(self.device)
            model = torch.nn.DataParallel(model)

            if os.path.isfile(self.trained_model_filepath):
                print(f'Loading pre-trained model {datetime.datetime.utcnow()}')
                trained_model_params = torch.load(self.trained_model_filepath)
                model.load_state_dict(trained_model_params)

            if self.is_training_run:
                print(f'''
                Training run
                Batching data {datetime.datetime.utcnow()}
                ''')
                data = nn_utils.batch_data(
                    x=x_pad,
                    y=y_pad,
                    n_batches=n_batches,
                )
                print(f'Fitting model {datetime.datetime.utcnow()}')
                nn_utils.fit(
                    model,
                    optimizer=self.optimizer(model),
                    loss_function=self.loss_function,
                    data=data,
                    input_shape=input_shape,
                    n_epochs=self.n_epochs,
                    device=self.device,
                )
                print(f'Saving model to {self.trained_model_filepath}: {datetime.datetime.utcnow()}')
                torch.save(model.state_dict(), self.trained_model_filepath)

            else:
                print(f'Generating prediction {datetime.datetime.utcnow()}')
                output = x_pad.copy()
                output['prediction'] = nn_utils.predict(
                    model=model,
                    x=x_pad,
                    n_batches=n_batches,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    device=self.device,
                )
                print(f'Post-processing data {datetime.datetime.utcnow()}')
                predictions = self.postprocess_data(input=input, output=output)
                if self.archive_files:
                    print(f'Saving model predictions to {self.location} {datetime.datetime.utcnow()}')
                    modeling_utils.save_file(
                        df=predictions,
                        subfolder=self.output_subfolder,
                        filename=self.filename(self.model_id),
                        is_prod=self.is_prod,
                    )

        else:
            print(f'Dataframe is empty. Check if data is missing: {self.start_date}')
