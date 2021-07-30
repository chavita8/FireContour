from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    numberContours = 3000
    segment.segmentImage(numberRays, numberContours)