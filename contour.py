import cv2

class Contour(object):
    def __init__(self, contour, color, growthMode):
        self.contour = contour
        self.color = color
        self.growthMode = growthMode

    def contourArea(self):
        area = cv2.contourArea(self.contour)
        return area