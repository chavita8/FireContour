from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from intersection import Intersection


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
        try:
            intersectionShape = polygon.exterior.intersection(self.toShapelyLine())
            centroide = self.originPoint
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
        except:
            print("except")
            #print(polygon)
        return self.intersectionsList

    def getDistances(self):
        for intersection in self.intersectionsList:
            distance = intersection.distance
            self.distancesList.append(distance)
        return self.distancesList