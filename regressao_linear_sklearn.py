# -*- coding: utf-8 -*-
#Created by Roberto Sá on 12/09/2021
#Algoritmo de regressão linear para informar o preço do do plano de saúde
#com base nos valores usando sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


#idade em anos
x = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]])

# custo em reais do plano de saúde conforme a idade
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],
              [1900]])


# visualizar gráfico de dispersão 
plt.scatter(x,y)

regressor = LinearRegression()
#treinamento
regressor.fit(x,y)
# y = b0 +b1*x
#b0
regressor.intercept_
print("Termo independente:" + str(regressor.intercept_))
#b1
regressor.coef_
print("Inclinação da reta: " + str(regressor.coef_))

#anos
idade = 40
previsao0 = regressor.intercept_ + regressor.coef_*idade
print("Testando primeiro modelo de previssão: " +  str(previsao0))

#forma automática passando um vetor

previsoes = regressor.predict(x)
print("Preço:\n" + str(previsoes))

#erro médio absoluto
mae = mean_absolute_error(y,previsoes)
print("Erro médio absoluto: " + str(mae))

#erro quadrado médio
mse = mean_squared_error(y,previsoes)
print("Erro quadrado médio: " + str(mse))  

plt.plot(x,y,'o')
plt.plot(x,previsoes, color = 'red')
plt.title("Modelo de regressão linear simples")
plt.xlabel('Idade')
plt.ylabel('Custo do plano')