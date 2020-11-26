from Segmentation import *
import cv2

def mostrarSegmentosPorImagenID(segmentos, imagenID):
    segmentosImagen = []
    for listaSegmento in segmentos:
        for segmento in listaSegmento:
            if segmento.imagenID == imagenID:
                print(segmento.imagenID)
                print(segmento.segmentoID)
                print(segmento.centroide)
                print(segmento.punto)
                print(segmento.distancia)
                segmentosImagen.append(segmento)

def mostrarSegmentosPorSegmentosID(segmentos, segmentoID):
    segmentosImagen = []
    for listaSegmento in segmentos:
        for segmento in listaSegmento:
            if segmento.segmentoID == segmentoID:
                print(segmento.imagenID)
                print(segmento.segmentoID)
                print(segmento.centroide)
                print(segmento.punto)
                print(segmento.distancia)
                segmentosImagen.append(segmento)
    return segmentosImagen

def calcularDiferenciaDistancia(segmentos, segmentoID):
    listaSegmentos = mostrarSegmentosPorSegmentosID(segmentos, segmentoID)
    i = 0
    print("------------------")
    print("Segmento: "+ str(segmentoID))
    while i<len(listaSegmentos):
        if i+1 <len(listaSegmentos):
            segmento1 = listaSegmentos[i]
            segmento2 = listaSegmentos[i+1]
            distancia1 = segmento1.distancia
            distancia2 = segmento2.distancia
            print("diferenciaDistancia:"+ str(segmento1.imagenID)+ " - "+ str(segmento2.imagenID))
            distancia = distancia2-distancia1
            print(distancia)
        i = i +1

if __name__ == '__main__':
    segmentacion = Segmentation()
    contador = 0
    segmentos = []
    cap = cv2.VideoCapture('video.mp4')
    if (cap.isOpened() == False):
        print("error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            contador = contador + 1;
            img, segmentos = segmentacion.segmentarImagen(frame, contador)
            cv2.imwrite("img" + str(contador) + ".png", img)
            #print("IMAGEN: "+ str(contador))
            cv2.imshow("detector", img)
            if cv2.waitKey(1) == 27:
                break
        else:
            break

    #mostrarSegmentosPorImagenID(segmentos, 1)
    #mostrarSegmentosPorSegmentosID(segmentos, 1)
    #segmentosImagenID = mostrarSegmentosPorSegmentosID(segmentos, 1)
    calcularDiferenciaDistancia(segmentos,4)
    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()