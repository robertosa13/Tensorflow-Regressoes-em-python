# -*- coding: utf-8 -*-
#Created by Roberto Sá on 12/09/2021
#Algoritmo de regressão linear para informar o preço do do plano de saúde
#com base nos valores usando sklearn

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#idade em anos
x = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]])

# custo em reais do plano de saúde conforme a idade
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],
              [1900]])

#valores escalonados, não altera o valor inicial
scaler_x = StandardScaler() #objeto scaler_x
x = scaler_x.fit_transform(x)  #recebe o objeto e transforma


scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

#fórmula de regressão linear simples
#y = b0+b1*x

np.random.seed(0) #semente geradora
a = np.random.rand(1)
b = np.random.rand(1)
a = a[0]
b = b[0]


b0 = tf.Variable(a)
b1 = tf.Variable(b)

erro = tf.losses.mean_squared_error(y, (b0 + b1 * x))
#print("Erro quadrado médio: " + str(erro))

#mínimo global utilizando derivadas parciais com a taxa de aprendizagem de 
#0,001 se a taxa for pequena demais pode demorar e se for grande demais pode
#perder o mínimo global que nesse problema indica o menor erro
#criei o gradiente
otimizador = tf.keras.optimizers.SGD(learning_rate=0.01)
#usando o menor erro
treinamento = otimizador.minimize(loss, var_list)

init = tf.global_variables_initializer()

