from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    time = 0.5
    samplingRate = 100
    oscillationFreq = 2
    segment.segmentImage(numberRays,time, samplingRate, oscillationFreq)