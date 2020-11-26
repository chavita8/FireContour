from Segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    numberContours = 5
    segment.segmentImage(numberRays,numberContours)