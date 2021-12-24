from segmentation import *

if __name__ == '__main__':
    segment = segmentation()
    numberRays = 20
    numberContours = 10
    segment.segmentImage(numberRays, numberContours)