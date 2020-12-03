from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from Intersection import Intersection

class Ray(object):
    def __init__(self, rayId, originPoint, destinationPoint):
        self.rayId = rayId
        self.originPoint = originPoint
        self.destinationPoint = destinationPoint
        self.direction = None
        self.intersectionList = []

    def rayId(self):
        return self.rayId

    def toShapelyLine(self):
        lineString = LineString([self.originPoint,self.destinationPoint])
        return lineString

    def intersect(self, polygon, id):
        intersectionShape = polygon.exterior.intersection(self.toShapelyLine())
        print("Intersect Method")
        print(type(intersectionShape))
        print(intersectionShape)
        if isinstance(intersectionShape, Point):
            intersection = Intersection(id, intersectionShape)
            self.intersectionList.append(intersection)
        if isinstance(intersectionShape, MultiLineString):
            print("Not Point")
            for line in intersectionShape:
                x, y = line.xy
                point1 = Point(x[0],y[0])
                point2 = Point(x[1],y[1])
                if point1.distance(point2) <= 4.0:
                    print("Distancia insignificante")
                    intersection1 = Intersection(id, point1)
                    self.intersectionList.append(intersection1)
                else:
                    intersection1 = Intersection(id, point1)
                    intersection2 = Intersection(id, point2)
                    self.intersectionList.append(intersection1)
                    self.intersectionList.append(intersection2)
                print(point1)
                print(point2)
            print("---------------")
        if isinstance(intersectionShape, MultiPoint):
            print("Not Point")
            for line in intersectionShape:
                x, y = line.xy
                point = Point(x[0],y[0])
                intersection = Intersection(id, point)
                self.intersectionList.append(intersection)
                print(point)
                print("---------------")
        return self.intersectionList







