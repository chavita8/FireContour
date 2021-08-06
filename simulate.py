from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    numberContours = 10
    segment.segmentImage(numberRays, numberContours)