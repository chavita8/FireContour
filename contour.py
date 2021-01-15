"""! @brief Definicion de la clase Contour. """

##
# @file contour.py
#
# @brief Define la clase Contour.
#
# @section description_contour Description
# Define la clase Contour que representa un contorno de la simulacion representado por un color y un modo de crecimiento: crece/decrece

import cv2

class Contour(object):
    """! clase Contour.
    Define la clase Contour que representa un contorno de la simulacion con su respectivo color y modo de crecimiento: crece/decrece """
    def __init__(self, contour, color, growthMode):
        """! constructor de la clase Contour
        @param contour Un contorno
        @param color Un color
        @param growthMode Modo de comportamiento: crece/decrece
        @return Una instancia de la clase Contour inicializada con los parametros: contour, color, growthMode
        """
        self.contour = contour
        self.color = color
        self.growthMode = growthMode

    def contourArea(self):
        """! metodo que calcula el area del contorno
        @return El area del contorno
        """
        area = cv2.contourArea(self.contour)
        return area