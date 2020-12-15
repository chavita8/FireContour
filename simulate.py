from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    time = 0.5
    sampling_rate = 100
    oscillation_freq = 1
    segment.segmentImage(numberRays,time, sampling_rate, oscillation_freq)