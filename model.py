import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from shapely.geometry import LineString
from shapely.wkt import loads
import cv2

class Model(object):
    def __init__(self):
        self.dataNumpy = None
        self.whiteColor = (255, 255, 255)
        self.blackColor = (0, 0, 0)
        self.blueColor = (255, 0, 0)
        self.greenColor = (0, 255, 0)
        self.redColor = (0, 0, 255)
        self.yellowColor = (0, 255, 255)
        self.cianColor = (255, 255, 0)
        self.pinkColor = (255, 0, 255)

    def predictLinear(self, data0, idRay, time_predict):
        data_x = data0['tiempo']
        data_y = data0['distancia']
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.33, random_state=1)

        x = train_x.to_numpy().reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, train_y)
        # print(model.coef_)
        # print(model.intercept_)

        test_x = test_x.to_numpy().reshape(-1, 1)
        pred = model.predict(test_x)
        xp = np.linspace(test_x.min(), test_x.max(), 70)
        xp = xp.reshape(-1, 1)
        pred_plot = model.predict(xp)

        title = "Lineal Ray " + str(idRay)
        scoreTest = model.score(test_x, test_y)
        print("train: ", model.score(train_x.to_numpy().reshape(-1, 1), train_y))
        print("test: ", model.score(test_x, test_y))
        plt.figure(figsize=(12, 6))
        plt.scatter(test_x, test_y, facecolor='None', edgecolor='k', alpha=0.3)
        plt.plot(xp, pred_plot)
        plt.xlabel("tiempo", fontsize=14)
        plt.ylabel("distancia", fontsize=14)
        plt.title(title, fontsize=18)
        plt.savefig("LinealRay" + str(idRay)+".png")

        size_data_x = len(data_x)
        new_time = [time_predict]
        test2 = np.array(new_time).reshape(-1, 1)
        predicted = model.predict(test2)
        print("DistanceNew Lineal :" + str(predicted))
        if scoreTest < 0.99:
            distancePredict = self.splinePredict(data0, idRay, time_predict)
            print("Spline DistanceNew : " + str(distancePredict))
        else:
            distancePredict = predicted[0]
        size_data_y = len(data_y)
        distancePoint = data_y[size_data_y - 1]
        # print("DistancePoint :"+ str(distancePoint))
        centroide_x = data0['centroide_x']
        centroide_y = data0['centroide_y']
        intersection_x = data0['intersection_x']
        intersection_y = data0['intersection_y']
        # print("PointCentroid :"+ str(pointCentroid))
        X0 = centroide_x[size_data_y - 1]
        Y0 = centroide_y[size_data_y - 1]
        print(X0)
        print(Y0)
        # print("X0 :"+ str(X0))
        # print("Y0 :"+ str(Y0))
        # print("PointFinal :"+ str(pointFinal))
        X1 = intersection_x[size_data_y - 1]
        Y1 = intersection_y[size_data_y - 1]
        # print("X1 :"+ str(X1))
        # print("Y1 :"+ str(Y1))
        # dPoint = calculateD(X0,Y0,X1,Y1)
        D = distancePredict / distancePoint
        # D = distancePredict/dPoint
        # print("D :"+ str(D))
        XD = ((1 - D) * X0) + (D * X1)
        # print("XD :"+ str(XD))
        YD = ((1 - D) * Y0) + (D * Y1)
        # print("YD :"+ str(YD))
        pointPredict = [XD, YD]
        print("PointPredict :" + str(pointPredict))
        return pointPredict

    def calculateD(self, X0, Y0, X1, Y1):
        distancia = math.sqrt((X1 - X0) ** 2 + (Y1 - Y0) ** 2)
        return distancia

    def splinePredict(self, df, idRay, time):
        df_x = df['tiempo']
        df_y = df['distancia']
        formula = ('df_y ~ bs(df_x, df=8, degree=1)')
        model_spline = smf.ols(formula=formula, data=df)
        result_spline = model_spline.fit()
        # print(result_spline.summary())
        times = []
        distances = []
        centroide_x = []
        centroide_y = []
        intersection_x = []
        intersection_y = []

        for i in range(0, len(df_x)):
            times.append(i)
            distances.append(0)
            centroide_x.append(0)
            centroide_y.append(0)
            intersection_x.append(0)
            intersection_y.append(0)
        time_dataframe = pd.DataFrame(list(zip(times, distances, centroide_x, centroide_y, intersection_x, intersection_y)), columns=['tiempo', 'distancia', 'centroide_x', 'centroide_y', 'intersection_x', 'intersection_y'])
        # print("DataFrameTime")
        # print(time_dataframe)
        datos = result_spline.predict(time_dataframe)
        # print("Result predict")
        # print(datos)
        datos_x = []
        datos_y = []
        for index, value in datos.items():
            datos_x.append(index)
            datos_y.append(value)
        title = "Spline Ray " + str(idRay)
        plt.figure(figsize=(12, 6))
        plt.scatter(datos_x, datos_y, facecolor='None', edgecolor='red', alpha=0.2)
        plt.xlabel("tiempo", fontsize=14)
        plt.ylabel("distancia", fontsize=14)
        plt.title(title, fontsize=18)
        plt.savefig("SplineRay" + str(idRay)+".png")

        newDistance = datos_y[time]
        return newDistance