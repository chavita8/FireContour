from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from intersection import Intersection
from model import Model
import numpy as np
import pandas as pd
import csv


class Ray(object):
    def __init__(self, rayId, originPoint, destinationPoint):
        self.rayId = rayId
        self.originPoint = originPoint
        self.destinationPoint = destinationPoint
        self.direction = None
        self.intersectionsList = []
        self.distancesList = []
        self.model = Model()

    def getRayId(self):
        return self.rayId

    def toShapelyLine(self):
        lineString = LineString([self.originPoint,self.destinationPoint])
        return lineString

    def intersect(self, polygon, id):
        #intersectionShape = polygon.exterior.intersection(self.toShapelyLine())
        lineString = self.toShapelyLine()
        intersectionShape = lineString.intersection(polygon)
        #print("INTERSECTION SHAPE")
        #print(type(intersectionShape))
        centroide = self.originPoint
        if isinstance(intersectionShape, LineString):
            if not intersectionShape.is_empty:
                coords = intersectionShape.coords;
                pointIni = coords[0]
                pointFin = coords[1]
                point1 = Point(int(pointIni[0]), int(pointIni[1]))
                point2 = Point(int(pointFin[0]), int(pointFin[1]))
                #print("point: ")
                #print(point2)
                distance = centroide.distance(point2)
                intersection = Intersection(id,intersectionShape,distance)
                self.intersectionsList.append(intersection)
        else:
            size_multiline = len(intersectionShape.geoms)
            lineStringIni = intersectionShape.geoms[0]
            pointIni = lineStringIni.coords[0]
            point1 = Point(int(pointIni[0]), int(pointIni[1]))
            lineStringFinal = intersectionShape.geoms[size_multiline-1]
            pointFin = lineStringFinal.coords[1]
            point2 = Point(int(pointFin[0]), int(pointFin[1]))
            distance = centroide.distance(point2)
            intersectionShape = LineString([point1, point2])
            intersection = Intersection(id,intersectionShape,distance)
            self.intersectionsList.append(intersection)
            """
            for intersection in intersectionShape.geoms:
                if isinstance(intersection, LineString):
                    if not intersection.is_empty:
                        coords = intersection.coords;
                        pointIni = coords[0]
                        pointFin = coords[1]
                        point1 = Point(int(pointIni[0]), int(pointIni[1]))
                        point2 = Point(int(pointFin[0]), int(pointFin[1]))
                        #print("point: ")
                        #print(point2)
                        distance = centroide.distance(point2)
                        intersection = Intersection(id, intersectionShape, distance)
                        self.intersectionsList.append(intersection)
            """

        """
        if isinstance(intersectionShape, Point):
            distance = centroide.distance(intersectionShape)
            intersection = Intersection(id, intersectionShape, distance)
            self.intersectionsList.append(intersection)
        if isinstance(intersectionShape, MultiLineString):
            x,y = intersectionShape[0].xy
            point = Point(x[1],y[1])
            distance = centroide.distance(point)
            intersection = Intersection(id, point, distance)
            self.intersectionsList.append(intersection)
        if isinstance(intersectionShape, MultiPoint):
            lenMultiPoint = len(intersectionShape)
            lastPoint = intersectionShape[lenMultiPoint-1]
            x,y = lastPoint.xy
            point = Point(x[0], y[0])
            distance = centroide.distance(point)
            intersection = Intersection(id, point, distance)
            self.intersectionsList.append(intersection)
        """
        return self.intersectionsList

    def getDistances(self):
        for intersection in self.intersectionsList:
            distance = intersection.distance
            self.distancesList.append(distance)
        return self.distancesList

    def getIntersections(self):
        return self.intersectionsList;

    def getDestinationPoint(self):
        return self.destinationPoint;

    def getOriginPoint(self):
        return self.originPoint;

    def predictPoint(self, time_predict):
        numpy_arr = self.generarDatos()
        #print(numpy_arr)
        len_numpy = len(numpy_arr)
        #print(numpy_arr)
        index_arr = list(range(len_numpy))
        #print(index_arr)
        df = pd.DataFrame(data=numpy_arr, index=index_arr, columns=["tiempo", "distancia", "centroide_x","centroide_y", "intersection_x","intersection_y"])
        #print(df)
        idRay = self.getRayId()
        print("---- RAY " + str(idRay)+" ----")
        point = self.model.predictLinear(df, idRay, time_predict)
        return point

    def generarDatos(self):
        #print('\n\n DISTANCES: ')
        self.getDistances()
        #print(len(self.distancesList))
        times = np.array(range(0, len(self.distancesList)))
        # times = np.linspace(0.1, 50.0, len(distances))
        X = times.reshape(-1, 1)
        Y = np.array(self.distancesList).reshape(-1, 1)
        csv_arr = []
        #csv_arr.append(["tiempo", "distancia", "centroide_x", "centroide_y", "intersection_x", "intersection_y"])
        for i, value in enumerate(X):
            if i < len(self.intersectionsList):
                arr = []
                x = X[i]
                y = Y[i]
                intersection = self.intersectionsList[i].intersectionPoint
                centroid = intersection.coords[0]
                centroid_x = centroid[0]
                centroid_y = centroid[1]
                pointIntersection = intersection.coords[1]
                intersection_x = pointIntersection[0]
                intersection_y = pointIntersection[1]
                arr.append(x[0])
                arr.append(y[0])
                arr.append(centroid_x)
                arr.append(centroid_y)
                arr.append(intersection_x)
                arr.append(intersection_y)

                csv_arr.append(arr)
        numpy_dataset = np.array(csv_arr)
        #filename = 'distanciasRayo' + str(self.rayId) + '.csv'
        #myFile = open(filename, 'w')
        #with myFile:
        #    writer = csv.writer(myFile)
        #    writer.writerows(csv_arr)
        return numpy_dataset
        # df = pd.read_csv(filename)s