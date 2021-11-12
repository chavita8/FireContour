from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from intersection import Intersection
import numpy as np
import csv
from shapely.wkt import loads

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
        #print("INTERSECTION SHAPE")
        #print(type(intersectionShape))
        centroide = self.originPoint
        sizeIntersections = 0
        currentIntersection = 0
        currentDistance = 0
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
                if distance > currentDistance:
                    intersectionObj = Intersection(id,intersectionShape,distance)
                    self.intersectionsList.append(intersectionObj)
                    sizeIntersections = len(self.intersectionsList)
                    currentIntersection = self.intersectionsList[sizeIntersections - 1]
                    currentDistance = currentIntersection.distance
        else:
            """
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
            sizeMultilines = len(intersectionShape.geoms)
            for intersection in intersectionShape.geoms:
                if isinstance(intersection, LineString):
                    if not intersection.is_empty:
                        lineString = intersectionShape.geoms[sizeMultilines - 1]
                        for i in range(2):
                            if i < len(lineString.coords):
                                coord = lineString.coords[i]
                                point = Point(int(coord[0]), int(coord[1]))
                                distance = centroide.distance(point)
                                if distance > currentDistance:
                                    lineStringStr = "LINESTRING ("+ str(centroide.x) + " "  + str(centroide.y) + ", " + str(point.x) + " " + str(point.y) + ")"
                                    lineString = loads(lineStringStr)
                                    intersectionObj = Intersection(id, lineString, distance)
                                    self.intersectionsList.append(intersectionObj)
                                    sizeIntersections = len(self.intersectionsList)
                                    currentIntersection = self.intersectionsList[sizeIntersections - 1]
                                    currentDistance = currentIntersection.distance


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
        print('\n\n DISTANCES CSV ' + str(self.rayId))
        print(len(self.intersectionsList))
        csv_arr = []
        csv_arr.append(["tiempo", "distancia","punto"])
        for intersection in self.intersectionsList:
            arr = []
            x = intersection.contourId
            y = intersection.distance
            point = intersection.intersectionPoint
            arr.append(x)
            arr.append(y)
            arr.append(point)
            csv_arr.append(arr)
        filename = 'distanciasRayo' + str(self.rayId) + '.csv'
        myFile = open(filename, 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(csv_arr)
        # df = pd.read_csv(filename)s