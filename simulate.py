from segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 8
    numberContours = 1
    segment.segmentImage(numberRays,numberContours)

    list = segment.obtenerSegmentosPorID(0)
    print("---------")
    for l in list:
        print(l.segmentoID)
        print(l.punto)