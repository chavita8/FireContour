from Segmentation import *
import cv2

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    numberContours = 5
    segment.segmentImage(numberRays,numberContours)