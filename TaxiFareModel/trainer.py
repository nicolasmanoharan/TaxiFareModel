from numpy.lib.shape_base import split
from sklearn.pipeline import Pipeline
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
from TaxiFareModel.dataa import get_data, clean_data


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
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
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self):
        pipe = self.set_pipeline().fit(self.X, self.y)
        return pipe




    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        temp =  self.run()
        y_pred = temp.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df, test=False)
    y = df.fare_amount
    X = df.drop(columns="fare_amount")
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    model = Trainer(X_train, y_train)
    #model.set_pipeline()
    #model.run()
    model.evaluate(X_test, y_test)
