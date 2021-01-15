"""! @brief Definicion de la clase Rayo. """

##
# @file ray.py
#
# @brief Define la clase Rayo.
#
# @section description_ray Description
# Define la clase Rayo que representa a todos los vectores trazados desde el centroide a las coordenadas polares definidas intersectando todos los contornos generados.


from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from intersection import Intersection
import math

class Ray(object):
    """! clase Rayo.
    Define la clase rayo utilizada por todos los vectores trazados desde el centroide hasta el ultimo contorno de la simulacion.
    """
    def __init__(self, rayId, originPoint, destinationPoint):
        """! constructor de la clase Rayo
        @param rayId El numero que identifica el rayo.
        @param originPoint El centroide que es el punto origen de todos los rayos.
        @param destinationPoint El punto destino del rayo en el ultimo contorno.
        @param direccion La direccion del vector rayo.
        @param intersectionsList La lista de intersecciones del rayo de cada contorno.
        @param distancesList La lista de distancias calculadas desde el centroide a cada punto de interseccion.
        @return Una instancia de la clase Rayo inicializada con los parametros: rayId, originPoint, destinationPoint
        """
        self.rayId = rayId
        self.originPoint = originPoint
        self.destinationPoint = destinationPoint
        self.direction = None
        self.intersectionsList = []
        self.distancesList = []

    def rayId(self):
        """! Obtiene el identificador del Rayo.
        @return El identificador del Rayo.
        """
        return self.rayId

    def toShapelyLine(self):
        """! Convierte el objeto Rayo a un objeto LineString de shapely para poder intersectar con los contornos de la simulacion.
        @return LineString Shapely
        """
        lineString = LineString([self.originPoint,self.destinationPoint])
        return lineString

    def intersect(self, polygon, id):
        """! Metodo de interseccion de un polygono (contorno) con el rayo
        @param polygon El objeto Polygon que representa un contorno de la simulacion.
        @param id El numero identificador del rayo al cual pertenece la interseccion.
        @return La lista de puntos de interseccion de todos los contornos que intersecta el rayo.
        """
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
        return self.intersectionsList

    def calcularDistances(self):
        """! Metodo de calculo de distancia desde el centroide a cada punto de interseccion del rayo
        @return La lista de distancias calculadas desde el centroide.
        """
        for intersection in self.intersectionsList:
            distance = intersection.calculateDistance(self.originPoint)
            self.distancesList.append(distance)
        return self.distancesList