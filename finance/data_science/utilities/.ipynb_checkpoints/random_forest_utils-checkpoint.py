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
                 objective='reg:linear',
                 columns_to_ignore=[]
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
        self.columns_to_ignore = columns_to_ignore

    # Data
    @property
    def train_x(self) -> pd.DataFrame:
        return self._train_x.drop(self.columns_to_ignore, axis=1)

    @property
    def train_y(self) -> pd.DataFrame:
        return self._train_y

    @property
    def test_x(self) -> pd.DataFrame:
        return self._test_x.drop(self.columns_to_ignore, axis=1)

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
        self.model = model

    def predict(self):
        self.prediction = self.model.predict(data=self.test_x)

    @property
    def mse(self):
        return metrics.mean_squared_error(y_pred=self.prediction, y_true=self.test_y)
    
    @property
    def output(self):
        prediction = pd.Series(self.prediction, name='prediction')
        df = pd.merge(self._test_x.reset_index(), 
                      prediction,
                      left_index=True,
                      right_index=True)
        return df
    
    def hyperparameter_optimization(self, 
                                    max_depths = [1, 2, 3, 5, 10],
                                    n_estimators = [10, 25, 50], #, 100, 200],
                                    gammas = [.01, .05, .1, .25, .5]
                                   ):
        mses = []
        max_depths = [1, 2, 3, 5, 10]
        for depth in max_depths:
            for estimator in n_estimators:
                for gamma in gammas:
                    self._max_depth = depth
                    self._n_estimators = estimator
                    self._gamma = gamma
                    
                    print(depth)
                    print(self.max_depth)
                    
                    self.fit()
                    self.predict()
                    
                    mse = self.mse
                    mses.append((depth, estimator, gamma, mse))
                    print('Finished fitting tree')
                    
        self.hyperparameter_mses = pd.DataFrame(mses, 
                                                columns=['max_depth', 
                                                         'n_estimators', 
                                                         'gamma', 
                                                         'mse'])

    def plot_prediction(self):
        plt.figure()
        plt.title('Random Forest Prediction versus Actuals')
        plt.plot(self.test_y, label='Actual')
        plt.plot(self.test_y.index, self.prediction, label='Predicted')
        plt.legend()
        plt.show()

    def plot_prediction_error(self):
        plt.figure()
        plt.title('Random Forest Prediction Errors')
        plt.plot(self.test_y.index, self.test_y - self.prediction)
        plt.hlines(0, xmin=self.test_y.index.min(), xmax=self.test_y.index.max())
        plt.show()
       
    def plot_prediction_groupby(self, 
                                groupby,
                                prediction='prediction',
                                actuals=None,
                                title='Prediction Groupby Plot', 
                                n_plots=10,
                                n_ticks=10,
                                error_plot=False,
                                df=None
                               ):
        n = 0
        df = self.output if df is None else df
        actuals = self.test_y.name if actuals is None else actuals
        
        plt.plot()
        for label, group in df.groupby(groupby):
            plt.title(title + ' ' + label)
            if not error_plot:
                plt.plot(group[prediction], 
                         label='Prediction ' + label)
                plt.plot(group[actuals],
                         label='Actual ' + label)
            else:
                plt.plot(group[prediction]-group[actuals], 
                         label='Error ' + label)
                plt.hlines(0, xmin=group.index.min(), xmax=group.index.max())
#             plt.xticks(ticks=group.index,
#                        labels=group['market_datetime'].dt.strftime('%m/%y'))
#             plt.locator_params(axis='x', nbins=n_ticks)
            plt.legend()
            plt.show()
            n += 1
            if n > n_plots:
                break

    def plot_tree(self, n_trees=2):
        xgb.plot_tree(self.model, num_trees=n_trees)

    def plot_importance(self, max_num_features=20):
        xgb.plot_importance(self.model, max_num_features=max_num_features)
        
    def plot_hyperparameters(self, measure='mean'):
        for param in ['max_depth', 'n_estimators', 'gamma']:
            if measure == 'mean':
                df = self.hyperparameter_mses.groupby(param).mean()
            elif measure == 'max':
                df = self.hyperparameter_mses.groupby(param).max()
            elif measure == 'min':
                df = self.hyperparameter_mses.groupby(param).min()
            plt.title(measure + ' mse over ' + param)
            plt.plot(df.index, df.mse)
            plt.show()
