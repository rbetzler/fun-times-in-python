import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt

from finance.utilities import utils


class XGBooster:
    def __init__(self,
                 train_x=None,
                 train_y=None,
                 test_x=None,
                 test_y=None,
                 max_depth=3,
                 learning_rate=1,
                 n_estimators=100,
                 verbosity=1,
                 booster='gbtree',
                 n_jobs=1,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 random_state=0,
                 objective='reg:linear'
                 ):
        # self._train_x = self.clean_data(train_x)
        self._train_x = train_x
        self._train_y = train_y
        # self._test_x = self.clean_data(test_x)
        self._test_x = test_x
        self._test_y = test_y
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._verbosity = verbosity
        self._booster = booster
        self._n_jobs = n_jobs
        self._gamma = gamma
        self._min_child_weight = min_child_weight
        self._max_delta_step = max_delta_step
        self._random_state = random_state
        self._objective = objective

    # Data
    @property
    def train_x(self) -> pd.DataFrame:
        return self._train_x

    @property
    def train_y(self) -> pd.DataFrame:
        return self._train_y

    @property
    def test_x(self) -> pd.DataFrame:
        return self._test_x

    @property
    def test_y(self) -> pd.DataFrame:
        return self._test_y

    @staticmethod
    def clean_data(df) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtypes is not int or float or bool:
                try:
                    df[col] = df[col].astype(float)
                except TypeError:
                    col + ' could not be coerced from ' + str(df[col].dtypes) + ' to float'
        return df

    # Tree params
    @property
    def max_depth(self) -> int:
        """
        max_depth (int) – Maximum tree depth for base learners.
        """
        return self._max_depth

    @property
    def learning_rate(self) -> float:
        """
        learning_rate (float) – Boosting learning rate (xgb’s “eta”)
        """
        return self._learning_rate

    @property
    def n_estimators(self) -> int:
        """
        n_estimators (int) – Number of trees to fit.
        """
        return self._n_estimators

    # Debugging
    @property
    def verbosity(self) -> int:
        """
        verbosity (int) – The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
        """
        return self._verbosity

    # Solver
    @property
    def booster(self) -> str:
        """
        booster (string) – Specify which booster to use: gbtree, gblinear or dart.
        """
        return self._booster

    @property
    def objective(self) -> str:
        """
        objective (string or callable) – Specify the learning task and the corresponding learning objective or a custom
        objective function to be used (see note below).
        """
        return self._objective

    @property
    def n_jobs(self) -> int:
        """
        n_jobs (int) – Number of parallel threads used to run xgboost. (replaces nthread)
        """
        return self._n_jobs

    # More params
    @property
    def gamma(self) -> float:
        """
        gamma (float) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
        """
        return self._gamma

    @property
    def min_child_weight(self) -> int:
        """
        min_child_weight (int) – Minimum sum of instance weight(hessian) needed in a child.
        """
        return self._min_child_weight

    @property
    def max_delta_step(self) -> int:
        """
        max_delta_step (int) – Maximum delta step we allow each tree’s weight estimation to be.
        """
        return self._max_delta_step

    @property
    def random_state(self) -> int:
        """
        random_state (int) – Random number seed. (replaces seed)
        """
        return self._random_state

    def fit(self):
        model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            booster=self.booster,
            objective=self.objective,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            random_state=self.random_state
        ).fit(X=self.train_x, y=self.train_y)
        return model

    def predict(self, model):
        return model.predict(data=self.test_x)

    def evaluate(self, prediction):
        df = metrics.mean_squared_error(y_pred=prediction, y_true=self.test_y)
        return df

    def plot_prediction(self, prediction):
        plt.figure()
        plt.title('Random Forest Prediction versus Actuals')
        plt.plot(self.test_y, label='Actual')
        plt.plot(self.test_y.index, prediction, label='Predicted')
        plt.legend()
        plt.show()

    def plot_prediction_error(self, prediction):
        plt.figure()
        plt.title('Random Forest Prediction Errors')
        plt.plot(self.test_y.index, self.test_y - prediction)
        plt.show()

    @staticmethod
    def plot_tree(model, n_trees=2):
        xgb.plot_tree(model, num_trees=n_trees)

    @staticmethod
    def plot_importance(model):
        xgb.plot_importance(model)


if __name__ == '__main__':
    query = """
        select
            extract(epoch from e.market_datetime) as market_datetime
            , e.open
            , e.high
            , e.low
            , e.close
            , e.volume
        from td.equities as e
        where e.symbol = 'BA'
        order by e.market_datetime
        """
    df = utils.query_db(query=query)

    temp = df[['market_datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    temp['market_datetime'] = temp['market_datetime'].astype(int)

    x = temp.drop('open', axis=1)
    y = temp['open'].shift(-1)

    train_x = x.iloc[1:1000]
    test_x = x.iloc[1010:1100]

    train_y = y.iloc[1:1000]
    test_y = y.iloc[1010:1100]

    boost = XGBooster(train_x=train_x, train_y=train_y,
                      test_x=test_x, test_y=test_y, max_depth=5)
    fit_model = boost.fit()
    prediction = boost.predict(model=fit_model)
    mse = boost.evaluate(prediction=prediction)
    print(test_y)
