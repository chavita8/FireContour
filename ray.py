from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from intersection import Intersection
import math

class Ray(object):
    def __init__(self, rayId, originPoint, destinationPoint):
        self.rayId = rayId
        self.originPoint = originPoint
        self.destinationPoint = destinationPoint
        self.direction = None
        self.intersectionList = []
        self.distanceList = []

    def rayId(self):
        return self.rayId

    def toShapelyLine(self):
        lineString = LineString([self.originPoint,self.destinationPoint])
        return lineString

    def intersect(self, polygon, id):
        intersectionShape = polygon.exterior.intersection(self.toShapelyLine())
        #print("Intersect Method")
        #print(type(intersectionShape))
        #print(intersectionShape)
        centroide = self.originPoint
        if isinstance(intersectionShape, Point):
            distance = centroide.distance(intersectionShape)
            intersection = Intersection(id, intersectionShape, distance)
            self.intersectionList.append(intersection)
        if isinstance(intersectionShape, MultiLineString):
            #print("Not Point")
            for line in intersectionShape:
                x, y = line.xy
                point1 = Point(x[0],y[0])
                point2 = Point(x[1],y[1])

                if point1.distance(point2) <= 4.0:
                    #print("Distancia insignificante")
                    distance = centroide.distance(point1)
                    intersection1 = Intersection(id, point1, distance)
                    self.intersectionList.append(intersection1)
                else:
                    distance1 = centroide.distance(point1)
                    distance2 = centroide.distance(point2)
                    intersection1 = Intersection(id, point1, distance1)
                    intersection2 = Intersection(id, point2, distance2)
                    self.intersectionList.append(intersection1)
                    self.intersectionList.append(intersection2)

                #print(point1)
                #print(point2)
            #print("---------------")
        if isinstance(intersectionShape, MultiPoint):
            #print("Not Point")
            for line in intersectionShape:
                x, y = line.xy
                point = Point(x[0],y[0])
                distance = centroide.distance(point)
                intersection = Intersection(id, point, distance)
                self.intersectionList.append(intersection)
                #print(point)
                #print("---------------")
        return self.intersectionList

    def calcularDistance(self):
        for intersection in self.intersectionList:
            intersectionPoint = intersection.intersectionPoint
            distance = intersection.calculateDistance(self.originPoint)
            print("CONTORNO")
            print(intersection.contourId)
            print("DISTANCIA")
            print(distance)
            self.distanceList.append(distance)
        return self.distanceList