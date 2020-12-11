from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    time = 0.3
    sampling_rate = 50
    oscillation_freq = 6.6
    segment.segmentImage(numberRays,time, sampling_rate, oscillation_freq)