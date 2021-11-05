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
                print("point: ")
                print(point2)
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
                        print("point: ")
                        print(point2)
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