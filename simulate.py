from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    numberContours = 50
    segment.segmentImage(numberRays, numberContours)