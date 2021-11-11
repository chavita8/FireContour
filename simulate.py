from segmentation import *

if __name__ == '__main__':
    segment = segmentation()
    numberRays = 20
    numberContours = 300
    segment.segmentImage(numberRays, numberContours)