import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def generadorDatosSimple(beta, muestras, desviacion):
    x = np.random.random(muestras)*100
    e = np.random.randn(muestras)*desviacion
    y = x*beta + e
    return x.reshape((muestras,1)), y.reshape((muestras,1))

redColor = (0, 0, 255)
greenColor = (0, 255, 0)

desviacion = 200
beta = 10
n = 50
x, y = generadorDatosSimple(beta, n, desviacion)
#plt.scatter(x, y)
#plt.show()

modelo = linear_model.LinearRegression()
modelo.fit(x, y)
#print("Coeficiente beta:"+ modelo.coef_[0])
yPred = modelo.predict(x)

#print("error cuadratico medio: "+ mean_squared_error(y, yPred))
#print("estadistico "+ r2_score(y, yPred))
plt.scatter(x, y)
plt.plot(x, yPred, redColor)
xReal = np.array([0, 100])
yReal = xReal*beta
plt.plot(xReal, yReal, greenColor)
plt.show()


#def linearRegression(self, rayList):
    #regresion_lineal = LinearRegression()
    # instruimos a la regresion lineal que aprenda de los datos (x,y)
    # x = np.arange(0,len(distances),1)

    # regresion_lineal.fit(times.reshape(-1, 1),distances)

    # vemos los parametros que ha estimado la regresion lineal
    #w = regresion_lineal.coef_
    #b = regresion_lineal.intercept_

    #print('w = ' + str(w))
    #print('b = ' + str(b))

    # vamos a predecir y = regresion_lineal(5)
    # nuevo_x = np.array([0])
    # prediccion = regresion_lineal.predict(times.reshape(-1, 1))
    # plt.scatter(times,prediccion)
    # print(prediccion)