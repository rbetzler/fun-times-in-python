from science.predictor import s1


class S3(s1.Predictor):
    @property
    def model_id(self) -> str:
        return 's3'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 500,
            'hidden_shape': 750,
            'dropout': 0.15,
            'learning_rate': .0001,
            'seed': 42,
            'sequence_length': 20,
            'batch_size': 10000,
        }
        return kwargs
