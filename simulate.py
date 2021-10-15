from segmentation import *

if __name__ == '__main__':
    segment = segmentation()
    numberRays = 20
    numberContours = 58
    segment.segmentImage(numberRays, numberContours)