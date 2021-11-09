from segmentation import *

if __name__ == '__main__':
    segment = segmentation()
    numberRays = 8
    numberContours = 1
    segment.segmentImage(numberRays, numberContours)