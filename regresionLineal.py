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
