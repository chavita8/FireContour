import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def func(x):
    N = 50  # number of data points
    f = 1.15247  # Optional!! Advised not to use
    data = 3.0 * np.sin(f * x + 0.001) + 0.5 + np.random.randn(N)  # create artificial data with noise
    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / (2 ** 0.5) / (2 ** 0.5)
    guess_phase = 0
    guess_freq = 1
    guess_amp = 1
    oscillationSine = guess_std * np.sin(x + guess_phase) + guess_mean
    return oscillationSine


N = 50  # number of data points
x = np.linspace(0, 2 * np.pi, N)
y = func(x)

plt.scatter(x, y, label='data', color='blue')
plt.title('Datos')
plt.show()

regresion_lineal = LinearRegression()
#instruimos a la regresion lineal que aprenda de los datos (x,y)
regresion_lineal.fit(x.reshape(-1,1),y)

#vemos los parametros que ha estimado la regresion lineal
w = regresion_lineal.coef_
b = regresion_lineal.intercept_

print('w = ' + str(w))
print('b = ' + str(b))

#vamos a predecir y = regresion_lineal(5)
nuevo_x = np.array([6.39429642])
prediccion = regresion_lineal.predict(nuevo_x.reshape(-1,1))
print("prediccion")
print(prediccion)

prediccion_entrenamiento = regresion_lineal.predict(x.reshape(-1,1))
#Calculamos el Error Cuadratico Medio (MSE = Mean Squared Error)
mse = mean_squared_error(y_true = y, y_pred= prediccion_entrenamiento)
#La raiz cuadrada del MSE es el RMSE
rmse = np.sqrt(mse)
print("Error Cuadratico Medio (MSE) = " + str(mse))
print("Raiz del Error Cuadratico Medio (RMSE) = " + str(rmse))

r2 = regresion_lineal.score(x.reshape(-1,1), y)
print('coeficiente de determinacion R2 = ' + str(r2))

plt.scatter(x,prediccion_entrenamiento, color="red")
plt.scatter(x,y,color="blue")
plt.show()