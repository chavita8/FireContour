from shapely.geometry.point import Point
from shapely.geometry import LineString
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
        point = polygon.exterior.intersection(self.toShapelyLine())
        intersection = Intersection(id, point)
        self.intersectionList.append(intersection)
        return self.intersectionList







