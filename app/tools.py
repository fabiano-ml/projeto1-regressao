# Imports
from datetime import datetime as dt
import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

## -----------------------------------------------------------------------------
def split_datetime(df):

	#converte tipo de data datetime64[ns, UTC] para datetime do pandas
	df['timestamp'] = pd.to_datetime(df.pickup_datetime).dt.tz_localize(None)

	#Separa data e hora os dados de treino
	df["hour"] = df.timestamp.dt.hour
	df["day_of_week"] = df.timestamp.dt.weekday
	df["day_of_month"] = df.timestamp.dt.day
	df["week"] = df.timestamp.dt.week
	df["month"] = df.timestamp.dt.month

	return df

## -----------------------------------------------------------------------------
def cyclic_features(df):

	#com as informacoes separadas, para preservar a informação cíclica.
	#ex: manter a aproximação (como em um círculo) dos valores, desta forma a hora 0 (meia-noite) está perto das 23 hr e 01 hr
	#e o dia 6 está perto do dia 0, etc...

	# executa a função cosseno após normalizar a coluna desejada entre 0 e 2π, o que corresponde a um ciclo de cosseno.
	# porém para não haver dois valores diferentes obteriam o mesmo valor cosseno deve-se calcular o ciclo seno.

	df["hour_norm"] = 2 * math.pi * df["hour"] / df["hour"].max()
	df["cos_hour"] = np.cos(df["hour_norm"])
	df["sin_hour"] = np.sin(df["hour_norm"])


	df["day_of_week_norm"] = 2 * math.pi * df["day_of_week"] / df["day_of_week"].max()
	df["cos_day_of_week"] = np.cos(df["day_of_week_norm"])
	df["sin_day_of_week"] = np.sin(df["day_of_week_norm"])

	df["day_of_month_norm"] = 2 * math.pi * df["day_of_month"] / df["day_of_month"].max()
	df["cos_day_of_month"] = np.cos(df["day_of_month_norm"])
	df["sin_day_of_month"] = np.sin(df["day_of_month_norm"])

	df["week_norm"] = 2 * math.pi * df["week"] / df["week"].max()
	df["cos_week"] = np.cos(df["week_norm"])
	df["sin_week"] = np.sin(df["week_norm"])

	df["hour_norm"] = 2 * math.pi * df["hour"] / df["hour"].max()
	df["cos_hour"] = np.cos(df["hour_norm"])
	df["sin_hour"] = np.sin(df["hour_norm"])


	return df	

## -----------------------------------------------------------------------------
def distance_km(lat1, lon1, lat2, lon2):
    ## conversão de graus decimais em radianos
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    ##  fórmula Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Raio da Terra em quilômetros. Use 3956 para milhas.
    
    return round(c * r,1)

def apply_distance(df):
    df['distance_km'] = distance_km(df.pickup_latitude, df.pickup_longitude,
                                df.dropoff_latitude, df.dropoff_longitude)
   
    return df

## -----------------------------------------------------------------------------

def mem_usage(pandas_obj):

	# espera-se que se user dataframe e não series

    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
        
    else: 
        usage_b = pandas_obj.memory_usage(deep=True)
        
    usage_mb = usage_b / 1024 ** 2 # Converte bytes para megabytes
    
    return "{:03.2f} MB".format(usage_mb)



def reduce_size_features(df, verbose):

	# reduz o tamanho das features ajustando os tipos
	if verbose:
		# mostra o uso de RAM atual do dataset antes da conversão
		antes = "Uso de RAM - antes da conversão : {}".format(mem_usage(df)) 
		print(antes + "\n")

	# avalia o benefício da redução que será aplicada
	df_int = df.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'int'])
	converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')

	if verbose:
		print("Uso de RAM das variáveis do int: {}".format(mem_usage(df_int)))
		print("Uso de RAM após conversão das variávies int: {} \n".format(mem_usage(converted_int)))

	df_float = df.select_dtypes(include=['float64', 'float32', 'float16', 'float'])
	converted_float = df_float.apply(pd.to_numeric,downcast='float')

	if verbose:
		print("Uso de RAM das variáveis do tipo float: {}".format(mem_usage(df_float)))
		print("Uso de RAM após conversão das variávies float: {} \n".format(mem_usage(converted_float)))
		print("Aplicando conversão dos tipos int e float ...\n")

	# aplica a conversão
	df[df_int.columns] = df_int.apply(pd.to_numeric, downcast='unsigned')
	df[df_float.columns] = df_float.apply(pd.to_numeric,downcast='float')


	# mostra o uso de RAM atual do dataset antes e após a conversão
	if verbose:
		print(antes)
		print("Uso de RAM - após a conversão: {} \n".format(mem_usage(df))) 

	return df

## -----------------------------------------------------------------------------
# Função para calcular o RMSE
def calc_rmse(modelo, X_train, y_train):
    rmse = np.sqrt(-cross_val_score(modelo, 
                                    X_train, 
                                    y_train, 
                                    scoring = "neg_mean_squared_error", 
                                    cv = 5))
    return(rmse)

def create_model_baseline(df, lst_X, Y, lst_ml):

    lst_rmse = []
    lst_r2 = []
    
    #pega o tamenho da lista de features
    n = len(lst_ml)
    #cria matriz de resultados
    
    # Criando um modelo de Regressão Linear
    modelo = LinearRegression(normalize = False, fit_intercept = True)

    #percorre toda lista de features e gera um modelo avaliado com as métricas RMSE e R2.
    for i in range(0,len(lst_X)):
        X = df[lst_X[i]]
        print('Criando modelo baseline - versão: ' + lst_ml[i] + ' ...')
        # Divisão em dados de treino e de teste
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        # Treinando o modelo
        modelo.fit(X_train, y_train)

        print('Calculando RMSE e R2 - versão: ' + lst_ml[i] + ' ...')
        #calcula métricas
        rmse = calc_rmse(modelo, X_train, y_train).mean()
        r2 = r2_score(y_test, modelo.fit(X_train, y_train).predict(X_test))
        
        #adiciona as métricas
        lst_rmse.append(rmse)
        lst_r2.append(r2)
        print('Versão ' + str(i) + ' - criada!\n')
        
    print('Concluída criação de modelo baseline !')
    
    return lst_rmse, lst_r2

## -----------------------------------------------------------------------------
def str_pd(s):
    dfx = pd.DataFrame()
    i = 0

    # faz split da string e coloca em um dataframe
    for ss in s.split(','):
        dfx.loc[0,i] = ss
        i += 1

    # ajusta os nome das colunas    
    dfx.set_axis(['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude'
                   ,'dropoff_latitude','passenger_count'], axis=1, inplace=True)

    # ajusta os tipos de dados das colunas
    dfx = dfx.astype({"pickup_datetime": 'datetime64[ns]'
    		,"pickup_longitude": 'float32'
    		,"pickup_latitude": 'float32'
    		,"dropoff_longitude": 'float32'
    		,"dropoff_longitude": 'float32'
    		,"dropoff_latitude": 'float32'
    		,"passenger_count":'int8'})

    return dfx


## -----------------------------------------------------------------------------
def feature_engineering(df, reduce):
	# transformando a feature datetime
	df = split_datetime(df)
	# convertendo para features cíclicas
	df = cyclic_features(df)
	# calcula distância em km
	df = apply_distance(df)
	
	if reduce:
		# reduz tamanho dos tipos das features numericas
		df = reduce_size_features(df, verbose=False)
	
	# remove features não necessárias
	df.drop(['key','pickup_datetime','timestamp'],axis=1, inplace=True) 

	return df

