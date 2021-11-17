from numpy.lib.shape_base import split
from sklearn.pipeline import Pipeline
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,ElasticNet
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
from TaxiFareModel.dataa import get_data, clean_data
from mlflow import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from joblib import dump
from TaxiFareModel.dataa import save_model



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[FR] [Paris] [nicolasmanoharan] model name + version"
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        dump(self.pipe, 'filename.joblib')

    pass
    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', ElasticNet())])
        return pipe

    def run(self):
        self.pipe = self.set_pipeline().fit(self.X, self.y)
        self.mlflow_log_param("model","Linear_model")
        return self.pipe




    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        temp =  self.run()
        y_pred = temp.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        self.mlflow_log_metric("rmse",rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()
# Nom de l'exper
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id
# Pousse les info sur MLflow
    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()

    df = clean_data(df, test=False)
    y = df.fare_amount
    X = df.drop(columns="fare_amount")
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    model = Trainer(X_train, y_train)
    #model.set_pipeline()
    #model.run()
    reg = model.evaluate(X_test, y_test)
    model.save_model(reg)
