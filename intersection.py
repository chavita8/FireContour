"""! @brief Definicion de la clase Intersection.  """

##
# @file intersection.py
#
# @brief Define la clase Intersection.
#
# @section description_intersection Description
# Define la clase Intersection que representa la interseccion de un rayo con un contorno.


class Intersection(object):
    """! clase Intersection
    Define la clase Intersection que representa un punto de interseccion de un rayo con un contorno.
    """
    def __init__(self, contourId, intersectionPoint, distance):
        """! constructor de la clase Intersection
        @param contourId El numero que identifica al contorno.
        @param intersectionPoint El punto de interseccion con el contorno.
        @param distance La distancia desde el centroide al punto de interseccion.
        @return Una instancia de la clase Intersection inicializada con los parametros: contourId, intersectionPoint, distance.
        """
        self.contourId = contourId
        self.intersectionPoint = intersectionPoint
        self.distance = distance

    def calculateDistance(self, centroide):
        """! metodo que realiza el calculo de la distancia desde el centroide hasta el punto de interseccion
        @param centroide El centroide
        @return Distancia calculada.
        """
        self.distance = centroide.distance(self.intersectionPoint)
        return self.distance