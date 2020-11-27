from Segmentation import *

if __name__ == '__main__':
    segment = Segmentation()
    numberRays = 9
    numberContours = 2
    segment.segmentImage(numberRays,numberContours)

    list = segment.obtenerSegmentosPorID(0)
    print("---------")
    for l in list:
        print(l.segmentoID)
        print(l.punto)