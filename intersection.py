
class Intersection(object):
    def __init__(self, contourId, intersectionPoint, distance):
        self.contourId = contourId
        self.intersectionPoint = intersectionPoint
        self.distance = distance

    def calculateDistance(self, centroide):
        self.distance = centroide.distance(self.intersectionPoint)
        return self.distance