from science.predictor import s1


class S2(s1.Predictor):
    @property
    def model_id(self) -> str:
        return 's2'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 500,
            'hidden_shape': 500,
            'dropout': 0.1,
            'learning_rate': .0001,
            'seed': 44,
            'sequence_length': 20,
            'batch_size': 1000,
        }
        return kwargs
