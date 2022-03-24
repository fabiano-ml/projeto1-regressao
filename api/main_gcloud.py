import sys

#inclui caminho de functions auxiliares
sys.path.append('../app')

from flask import Flask
import pandas as pd
import pickle
import warnings
import tools
warnings.filterwarnings("ignore")

modelo = pickle.load(open('../modelo/modelo_20220321.sav','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "OK - home - API."

@app.route('/pred_batch/<path:input_url>')
def pred_batch(input_url):
    
    # Ex: http://127.0.0.1:5000/pred_batch/d:/serasa/data/test.csv
        
    df_test = pd.read_csv(input_url, parse_dates=["pickup_datetime"])
    df = df_test.copy()
    df = tools.feature_engineering(df,reduce=True)

    # faz predicao
    # acrescenta o resultado da predição do dataset de test
    df_test['pred_fare_amount'] = modelo.predict(df)
       
    # salva dataset com valor achado de predição ao dataset de test
    path_out = "../data/test_pred_batch.csv"
    df_test.to_csv(path_out);
    
    return "Arquivo entrada: " + input_url + " - arquivo saída: " + path_out

@app.route('/pred_online/<sinput>')
def pred_online(sinput):

    #Ex: http://127.0.0.1:5000/pred_online/2015-01-27 13:08:24.0000002,2015-01-27 13:08:24 UTC,-73.973320007324219,40.7638053894043,-73.981430053710938,40.74383544921875,1
        
    # pega a entrada e transforma em um dataframe
    df = tools.feature_engineering(tools.str_pd(sinput),reduce=False)
    
    # faz predicao
    pred = modelo.predict(df)
    pred_fare_amount = pred[0]
    
    return "pred_fare_amount - " + str(pred_fare_amount)

# garantir que seja esecutada  pelo nome.
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')    
