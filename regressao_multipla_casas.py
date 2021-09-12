# -*- coding: utf-8 -*-
#Created by Roberto Sá on 12/09/2021
#Regressão múltipla para precificação de imóveis


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_estimator
import tensorflow_estimator.python.estimator.api._v2.estimator
import tensorflow_core.estimator 
from sklearn.metrics import mean_absolute_error
import numpy as np

#importa arquivo CSV (Colunas separadas por vírgulas)
base = pd.read_csv('precos-casas.csv')


#inserir as colunas utilizadas para a análise conforme arquivo CSV
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
#se uma coluna não tiver acima, não vai ser adicionada
base = pd.read_csv('precos-casas.csv', usecols = colunas_usadas)


#normalização

scaler_x = MinMaxScaler() # criação do objeto MinMaxScaler

#retirei algumas colunas daqui e com esse comando acesso cada coluna abaixo
#dados normalizados entre 0 e 1
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])
                                                      
#quero determinar o preço                                              
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])



X = base.drop('price', axis = 1) #apagar a coluna inteira e não mexe na base
y = base.price #recebeu apenas a coluna


previsores_colunas = colunas_usadas[1:17] #posicao 1 até 17, posicao 0 é o preco
# e não quero puxar essa coluna
previsores_colunas    

#percorrendo cada índice
colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

#divisão base de dados de teste e treinamento
#70% para treinar e 30% para testar

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3)

#data --> input fuction -->estimator (Train,Evaluate, Predict)

#função de treinamento
#shuffle para misturar os dados 

funcao_treinamento = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = X_treinamento, y = y_treinamento, batch_size = 32, 
    num_epochs = None, shuffle = True) 

funcao_teste = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = X_teste, y = y_teste, batch_size = 32, 
    num_epochs = 10000, shuffle = False)  


regressor = tf.estimator.LinearRegressor(feature_columns=colunas)

regressor.train(input_fn=funcao_treinamento, steps = 10000)
metricas_treinamento = regressor.evaluate(input_fn=funcao_treinamento, steps = 10000)


#teste
metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps = 10000)                   

metricas_treinamento
metricas_teste

#função de previsão
funcao_previsao = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_teste, shuffle = False)

valores_previsoes = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsoes.append(p['predictions'])
    
    
valores_previsoes
#transformar em numpy array para facilitar
valores_previsoes = np.asarray(valores_previsoes).reshape(-1,1)
valores_previsoes = scaler_y.inverse_transform(valores_previsoes)


#desnormalizar o y teste
y_teste2 = y_teste.values.reshape(-1,1)


y_teste2 = scaler_y.inverse_transform(y_teste2)
y_teste2

#erro médio absoluto
mae = mean_absolute_error(y_teste2, valores_previsoes)

print(mae)
#123375.40887476587








