import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

class Model:
    def __init__(self):
        uri = "mongodb+srv://skansal291:lLmDtxj4v4J7nkku@clusterml.qr74icy.mongodb.net/?retryWrites=true&w=majority&appName=Clusterml"
        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))
        # connect to database
        db = client.get_database('carprediction')
        # connect to the collection
        self.records = db['salesprediction']
        # self.df = pd.DataFrame(list(self.records.find({})))
        # self.df.drop(['_id'], axis=1, inplace=True)
        self.df = pd.read_csv('CarSeats.csv')
    def train(self):
        df1 = self.df.drop(columns=['No'])
        x = df1.drop(columns=['Sales'])
        y = df1['Sales']
        categorical_cols = x.select_dtypes(include=['object','category']).columns.tolist()
        numerical_cols = x.select_dtypes(include=['number']).columns.tolist()
        preprocessor = ColumnTransformer([('nums',StandardScaler(), numerical_cols),('cat',OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
        pipeline = Pipeline([('preprocessor',preprocessor),('mlp',MLPRegressor(hidden_layer_sizes=(100,),activation='relu',solver='adam',learning_rate_init=1e-3,max_iter=200,random_state=42))])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        pipeline.fit(x_train,y_train)
        joblib.dump(pipeline,'carseates.pkl')
        loaded_pipeline = joblib.load('carseates.pkl')
        y_pred = loaded_pipeline.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)
        return r2
    
if __name__ == '__main__':
    m = Model()
    s = m.train()
    print("R2 score:", str(s))
    

