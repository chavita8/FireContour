from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from intersection import Intersection
import numpy as np
import csv


class Ray(object):
    def __init__(self, rayId, originPoint, destinationPoint):
        self.rayId = rayId
        self.originPoint = originPoint
        self.destinationPoint = destinationPoint
        self.direction = None
        self.intersectionsList = []
        self.distancesList = []

    def rayId(self):
        return self.rayId

    def toShapelyLine(self):
        lineString = LineString([self.originPoint,self.destinationPoint])
        return lineString

    def intersect(self, polygon, id):
        #intersectionShape = polygon.exterior.intersection(self.toShapelyLine())
        intersectionShape = self.toShapelyLine().intersection(polygon)
        print("intersection line")
        print(intersectionShape)
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

    def generarCSV(self):
        print('\n\n DISTANCES CSV: ')
        self.getDistances()
        print(len(self.distancesList))
        times = np.array(range(0, len(self.distancesList)))
        # times = np.linspace(0.1, 50.0, len(distances))
        X = times.reshape(-1, 1)
        Y = np.array(self.distancesList).reshape(-1, 1)
        csv_arr = []
        csv_arr.append(["tiempo", "distancia","punto"])
        for i, value in enumerate(X):
            arr = []
            x = X[i]
            y = Y[i]
            intersection = self.intersectionsList[i].intersectionPoint
            arr.append(x[0])
            arr.append(y[0])
            arr.append(intersection)
            csv_arr.append(arr)
        filename = 'distanciasRayo' + str(self.rayId) + '.csv'
        myFile = open(filename, 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(csv_arr)
        # df = pd.read_csv(filename)s