from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score 

from training import Model as m
from database import Database as D
#load model
model = joblib.load('carseates.pkl')
mdl = m()
df = mdl.df
dtbs = D()
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template("simple_form.html")

@app.route('/predict',methods=['GET','POST'])
def Prediction():
    input_data=None
    prediction=None
    features=[]
    prediction=""
    if request.method == 'POST':
        input_data = request.form.to_dict()
        #retrieve form values and convert to floats
        for x in request.form.values():
            features.append(x)
        columns_names = ['CompPrice', 'Income', 'Advertising','Population', 'Price', 'Age', 'Education', 'ShelveLoc', 'Urban', 'US']
        x_new = pd.DataFrame([features], columns = columns_names)
        #prediction
        prediction = model.predict(x_new)
        return render_template('predict.html', input_data=input_data,prediction=int(prediction[0]))
    else:
        return render_template('predict.html',input_data=input_data,prediction=prediction)

@app.route('/eda',methods=['GET','Post'])
def eda():
    image_file = None
    selected_x = None
    selected_y = None
    selected_plot = None
    title = ''
    df=pd.read_csv('CarSeats.csv')
    columns = df.columns.tolist()
    if request.method == 'POST':
        selected_x = request.form.get('xcolumn')
        selected_y = request.form.get('ycolumn')
        selected_plot = request.form.get('plottype')
        if selected_plot == 'bar':
            plt.bar(df[selected_x], df[selected_y],color='salmon')
            title = f'Bar chart {selected_x} Vs {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.grid(True)
            plt.savefig("static/images/bar.png")
            plt.close()
            image_file = 'images/bar.png'
        elif selected_plot == 'line':
            df.sort_values(by=selected_x, inplace=True)
            avg=df.groupby(selected_x)[selected_y].mean()
            avgv=avg.values
            avgi=avg.index
            plt.plot(avgi, avgv, marker='o',linestyle='--', color='lightgreen')
            title = f'Line plot {selected_x} across {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.grid(True)
            plt.savefig("static/images/line.png")
            plt.close()
            image_file = 'images/line.png'
        elif selected_plot == 'scatter':
            df.sort_values(by=selected_x, inplace=True)
            plt.scatter(df[selected_x], df[selected_y], color='teal')
            title = f'Scatter plot {selected_x} across {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.grid(True)
            plt.savefig("static/images/scatter.png")
            plt.close()
            image_file = 'images/scatter.png'
        elif selected_plot == 'hbar':
            plt.barh(df[selected_x], df[selected_y], color='pink')
            title = f'Horizontal Bar Chart {selected_x} vs {selected_y}'
            plt.title(title)
            plt.xlabel(selected_y)
            plt.ylabel(selected_x)
            plt.grid(True)
            plt.savefig("static/images/hbar.png")
            plt.close()
            image_file = 'images/hbar.png'
        elif selected_plot == 'hist':
            plt.hist(df[selected_x], bins=5, color='skyblue',edgecolor='black')
            title = f'Histogram {selected_x} Vs Frequency'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel('frequency')
            plt.savefig("static/images/hist.png")
            plt.close()
            image_file = 'images/hist.png'
        elif selected_plot == 'pie':
            data = df[selected_x].value_counts()
            labels=data.index
            sizes=data.values
            plt.pie(sizes,labels=labels, autopct='%.2f%%',startangle=90)
            title = f'Pie chart {selected_x}'
            plt.title(title)
            plt.savefig("static/images/pie.png")
            plt.close()
            image_file = 'images/pie.png'
    return render_template('EDA.html',columns=columns, image_file=image_file)

@app.route('/train',methods=['GET','POST'])
def train_model():
    s = None
    if request.method == 'POST':
        s = mdl.train()
    return render_template('train.html',score=s)

@app.route('/add_data', methods = ['GET','POST'])
def add_data():
    columns = list(zip(df.columns, df.dtypes))
    input_data = {}
    result = None
    message = None
    data = None
    if request.method == 'POST':
        input_data = request.form.to_dict()
        for name, dtype in columns:
            if dtype in ['int']:
                input_data[name] = int(input_data[name])
            elif dtype in ['float']:
                input_data[name] = float(input_data[name])
        result = dtbs.add_single_document(input_data)
        
        if result:
            message = 'Insertion Successful'
            print(message)
        data = list(dtbs.records.find())
        for doc in data:
            doc.pop('_id',None)
            
    return render_template('add_data.html', columns=columns, message=message, data=data)

@app.route('/cluster', methods = ['GET','POST'])
def cluster():
    
    image_file = None
    column1 = None
    column2 = None
    x = None
    scaler = None
    df=pd.read_csv('CarSeats.csv')
    columns = df.select_dtypes(include=['number']).columns.tolist()
    if request.method == 'POST':
        
        column1 = request.form.get('column1')
        column2 = request.form.get('column2')
        x = df[[column1,column2]].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        wcss = []
        k_range = range(1,11)
        for k in k_range:
            km = KMeans(n_clusters=k,random_state=42,n_init=12)
            km.fit(x_scaled)
            wcss.append(km.inertia_)
        plt.figure(figsize=(6,4))
        plt.plot(k_range,wcss,marker='o')
        plt.xlabel('NUMBER OF CLUSTER(K)')
        plt.ylabel('WCSS')
        plt.title("number of clusters vs wcss")
        plt.xticks(k_range)
        plt.savefig("static/images/k.png")
        plt.close()
        image_file = 'images/k.png'
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42,n_init=12)
        labels = kmeans.fit_predict(x_scaled)
        df['cluster'] = labels
        print(df['cluster'].value_counts())
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)
        plt.figure(figsize=(10,8))
        for i in range(optimal_k):
            plt.scatter( x[labels == i,0], x[labels == i,1], label = f'cluster{i}')
        plt.scatter(centroids[:,0], centroids[:,1], color='black',marker='X', label = 'centroids')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f"cluster between:{column1} VS {column2}")
        plt.legend()
        plt.savefig("static/images/cluster.png")
        plt.close()
        image_file = 'images/cluster.png'
    return render_template('cluster.html',columns=columns, image_file=image_file)  

@app.route('/pca',methods=['GET','POST'])
def pca():
    image_file= None
    mse = None
    r2 = None
    loaded_r2 = None
    df = pd.read_csv('CarSeats.csv')
    df = df.drop(columns=['No'])
    if request.method == 'POST':
        
        x = df.drop(columns=['Sales'])
        y = df['Sales']
        categorical_cols = x.select_dtypes(include=['object','category']).columns.tolist()
        numerical_cols = x.select_dtypes(include=['number']).columns.tolist()
        preprocessor = ColumnTransformer([('nums',StandardScaler(), numerical_cols),('cat',OneHotEncoder(handle_unknown='ignore'),categorical_cols)])
        pca = PCA(n_components=14, random_state=42)
        pipeline = Pipeline([('preprocessor', preprocessor),('pca',pca),('mlp',MLPRegressor(hidden_layer_sizes=(100,),activation='relu',solver='adam',learning_rate_init=1e-3,max_iter=200,random_state=42))])
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
        pipeline.fit(x_train,y_train)
        explained = pca.explained_variance_ratio_
        var_df = pd.DataFrame({'principle component':np.arange(1,len(explained)+1), 'explained variance ratio':explained})
        plt.figure(figsize=(12,10))
        plt.bar(var_df['principle component'],var_df['explained variance ratio'],label='individual')
        plt.xlabel('Principle Component',fontsize=12)
        plt.ylabel('Explained Variance Ratio',fontsize=12)
        plt.title('Explained Variance Ratio by Principle Component',fontsize=14)
        plt.legend()
        plt.xticks(np.arange(1,len(explained)+1))
        plt.savefig('static/images/pca.png')
        plt.close()
        image_file = 'images/pca.png'
        y_pred = pipeline.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)
        joblib.dump(pipeline, 'carseates.pkl')
        loaded = joblib.load('carseates.pkl')
        loaded_r2 = r2_score(y_test, loaded.predict(x_test))
    return render_template('PCA.html', image_file=image_file, mse=mse, r2=r2, loaded_r2=loaded_r2)

@app.route('/cv', methods=['GET','POST'])
def cv():
    image_file = None
    mean = None
    cv_scores = None
    if request.method == 'POST':
        df = pd.read_csv('CarSeats.csv')
        df = df.drop(columns=['No'])
        x = df.drop(columns=['Sales'])
        y = df['Sales']
        categorical_cols = x.select_dtypes(include=['object','category']).columns.tolist()
        numerical_cols = x.select_dtypes(include=['number']).columns.tolist()
        preprocessor = ColumnTransformer([('nums',StandardScaler(), numerical_cols),('cats',OneHotEncoder(handle_unknown='ignore'),categorical_cols)])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),('mlp', MLPRegressor(hidden_layer_sizes=(100,),activation='relu',solver='adam',learning_rate_init=1e-3,max_iter=200, random_state=42))
        ])
        cv_scores = cross_val_score(pipeline, x,y, cv=12, n_jobs=-1)
        folds = np.arange(1, len(cv_scores)+1)
        plt.figure(figsize=(15,12))
        plt.bar(folds, cv_scores, color='skyblue')
        plt.xlabel('FOLDS',fontsize=12)
        plt.ylabel('R2_SCORE',fontsize=12)
        plt.title('Cross Validation Scores',fontsize=14)
        plt.ylim(cv_scores.min()-0.008, cv_scores.max()+0.008)
        plt.xticks(folds)
        plt.savefig('static/images/cv.png')
        plt.close()
        image_file = 'images/cv.png'
        mean = cv_scores.mean()
    return render_template('cv.html', image_file=image_file, cv_scores=cv_scores, mean=mean)

@app.route('/reg', methods=['GET','POST'])
def reg():
    r2_ridge = None
    r2_lasso = None
    if request.method == 'POST':
        df = pd.read_csv('CarSeats.csv')
        df = df.drop(columns=['No'])
        df = df.select_dtypes(include='number')
        x = df.drop(columns=['Sales'])
        y = df['Sales']
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)
        pipeline_ridge = Pipeline([
            ('nums', StandardScaler()), ('regressor',SGDRegressor(max_iter=1000, learning_rate='adaptive',eta0=0.01,penalty='l2',alpha=0.01))
        ])
        pipeline_ridge.fit(x_train,y_train)
        y_pred_ridge = pipeline_ridge.predict(x_test)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        pipeline_lasso = Pipeline([
            ('nums', StandardScaler()), ('regressor',SGDRegressor(max_iter=1000, learning_rate='adaptive',eta0=0.01,penalty='l1',alpha=0.01))
        ])
        pipeline_lasso.fit(x_train,y_train)
        y_pred_lasso = pipeline_lasso.predict(x_test)
        r2_lasso = r2_score(y_test, y_pred_lasso)
    return render_template('reg.html',r2_ridge=r2_ridge, r2_lasso=r2_lasso)

@app.route('/rnn', methods=['GET','POST'])
def rnn():
    image_file = None
    columns = None
    title = ''
    df = pd.read_csv('CarSeats.csv')
    df = df.drop(columns=['No'])
    df= df.select_dtypes(include='number')
    columns = df.columns.tolist()
    if request.method == 'POST':
        columns = request.form.get('column')
        features = [columns]
        train_df, test_df = train_test_split(df[features],test_size=0.3, random_state=42)
        train_vals = train_df.values
        test_vals = test_df.values
        mins = train_vals.min(axis=0)
        maxs = train_vals.max(axis=0)
        train_norm = (train_vals - mins) / (maxs - mins)
        test_norm = (test_vals - mins) / (maxs - mins)
        def create_sequences(data,window_size=30):
            x, y = [],[]
            for i in range(len(data)-window_size):
                x.append(data[i:i+window_size])
                y.append(data[i+window_size])
            return np.array(x),np.array(y)
        window_size = 30
        x_train, y_train = create_sequences(train_norm, window_size)
        x_test, y_test = create_sequences(test_norm, window_size)
        num_features = len(features)
        """ model = tf.keras.Sequential([
            layers.SimpleRNN(50, input_shape=(window_size, num_features), activation='tanh'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train,y_train, epochs=50, batch_size=32, validation_split=0.2) """
        model = load_model('rnn_model.keras')
        print(model.evaluate(x_test,y_test))
        true_c = y_test * (maxs - mins) + mins
        prediction_norm = model.predict(x_test)
        pred_c = prediction_norm * (maxs - mins) + mins
        plt.figure(figsize=(12,6))
        plt.plot(true_c, label='True Values', color='blue')
        plt.plot(pred_c, label='Predicted Values', color='orange')
        plt.xlabel('Sample Index',fontsize=14)
        plt.ylabel(columns,fontsize=14)
        title = f'RNN Prediction for {columns}'
        plt.title(title, fontsize=16)
        plt.legend()
        plt.savefig('static/images/rnn.png')
        plt.close()
        image_file = 'images/rnn.png'
    return render_template('rnn.html', image_file=image_file, columns=columns)
  
@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    image_file = None
    sentiment = None
    df = pd.read_csv('sentiment_analysis.csv')
    if request.method == 'POST':
        comment = request.form.get('comment')
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        def clean_text(s):
            s = s.lower()
            filtered = ''.join(ch if( ch.isalpha() or ch.isspace()) else ' ' for ch in s)
            return ' '.join(filtered.split())
        train_df['clean_text'] = train_df['text'].astype(str).map(clean_text)
        val_df['clean_text'] = val_df['text'].astype(str).map(clean_text)
        train_df = train_df[train_df['clean_text'].str.strip() != ''].reset_index(drop=True)
        val_df = val_df[val_df['clean_text'].str.strip() != ''].reset_index(drop=True)
        train_texts = train_df['clean_text'].tolist()
        val_texts = val_df['clean_text'].tolist()
        labels  = sorted(train_df['sentiment'].unique())
        label_to_index = {label:i for i,label in enumerate(labels)}
        train_idx = np.array([label_to_index[s] for s in train_df['sentiment']])
        val_idx = np.array([label_to_index[s] for s in val_df['sentiment']])
        num_classes = len(labels)
        train_labels = to_categorical(train_idx, num_classes)
        val_labels = to_categorical(val_idx, num_classes)
        MAX_WORDS = 10000
        MAX_LEN = 100
        tokenizer = Tokenizer(num_words = MAX_WORDS)
        tokenizer.fit_on_texts(train_texts)
        """ model = Sequential([
            Embedding(MAX_WORDS,128, input_length=MAX_LEN),
            Bidirectional(LSTM(64, dropout=0.2 ,recurrent_dropout=0.2)),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics = ['accuracy'])
        model.fit(pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post', truncating='post'), train_labels, epochs=25, batch_size=32,
            validation_data = (pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN, padding='post', truncating='post'), val_labels),verbose=1) """
        model = load_model('sentiment_model.keras')
        def clean_text(s):
            s = s.lower()
            filtered = ''.join(ch if( ch.isalpha() or ch.isspace()) else ' ' for ch in s)
            return ' '.join(filtered.split())
        
        def predict_sentiment(comment):
            cleaned = clean_text(comment)
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen = MAX_LEN, padding='post', truncating='post')
            probs = model.predict(pad)
            idx = np.argmax(probs, axis=1)[0]
            return labels[idx], probs[0]
        sentiment, _ = predict_sentiment(comment)
        if sentiment == 'positive':
            image_file = 'images/thumbs_up.png'
        elif sentiment == 'negative':
            image_file = 'images/thumbs_down.png'
        elif sentiment == 'neutral':
            image_file = 'images/nutral.png'

    return render_template('sentiment.html', sentiment=sentiment, image_file = image_file)
                
            
    
if __name__ == '__main__':
    app.run(debug=True)
    


