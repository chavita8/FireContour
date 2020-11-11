import cv2
import numpy as np
import math
from shapely.wkt import loads
from shapely.geometry.point import Point
from filtros.Segmento import Segmento


class Segmentacion(object):
    def __init__(self):
        self.imagen = cv2.imread("fuego1.png")
        self.colorAzul = (255, 0, 0)
        self.colorVerde = (0, 255, 0)
        self.colorRojo = (0, 0, 255)
        self.colorAmarillo = (0, 255, 255)
        self.colorRosa = (255, 255, 0)
        self.centroides = []
        self.invariantesMomentosHu = []
        self.invariantesSiftPoints = []
        self.invariantesSurtPoints = []
        self.segmentos = []

    def segmentarImagen(self, frame, imagenID):
        if np.any(frame):
            #imagenRuido = cv2.medianBlur(self.imagen, 11)
            imagenRuido = cv2.medianBlur(frame, 11)
            imagenHSV = cv2.cvtColor(imagenRuido, cv2.COLOR_BGR2HSV)
            canalHSV = cv2.split(imagenHSV)
            (canalH, canalS, canalV) = canalHSV
            cannyImage = cv2.Canny(canalV, 127, 255)
            cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentar/Segmentacion/filtros/canny.png", cannyImage)

            imagenRojoAmarillo = self.segmentarRojoAmarrillo(imagenHSV)
            cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentar/Segmentacion/filtros/mascaraRoja.png", imagenRojoAmarillo)
            imagenVerde = self.segmentarVerde(imagenHSV)
            cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentar/Segmentacion/filtros/mascaraVerde.png", imagenVerde)
            cannyImageRojo = cv2.Canny(imagenRojoAmarillo, 127, 255)
            cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentar/Segmentacion/filtros/cannyRojo.png", cannyImageRojo)

            #dilateImage = self.aplicarDilate(cannyImageRojo)
            #cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentacion/filtros/dilateCanny.png", dilateImage)
            _, contornos, _ = cv2.findContours(cannyImageRojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #_, contornos, _ = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print("--- contornos ---")
            #print (type(contornos))
            #print(len(contornos))
            #cv2.drawContours(self.imagen,contornos,-1, self.colorAzul, 3)
            #print(contornos)
            for c in contornos:
                centroide = self.encontrarCentroide(c)
                cv2.circle(self.imagen, centroide, 3, self.colorRosa, 3)
                cv2.circle(frame, centroide, 3, self.colorRosa, 3)
                #print("---  Centroide  ---")
                #print(centroide)
            self.centroides.append(centroide)

            #cv2.line(self.imagen, (int(centroide[0])-200,int(centroide[1])),(int(centroide[0])+200,int(centroide[1])), self.colorAzul,1)
            cv2.line(frame, (int(centroide[0])-200,int(centroide[1])),(int(centroide[0])+200,int(centroide[1])), self.colorAzul,1)
            cv2.line(frame, (int(centroide[0]),int(centroide[1]-200)),(int(centroide[0]),int(centroide[1])+200), self.colorAzul,1)
            cv2.line(frame, (int(centroide[0]-200), int(centroide[1]-200)),(int(centroide[0]+200),int(centroide[1])+200), self.colorAzul,1)
            cv2.line(frame, (int(centroide[0]+200), int(centroide[1]-200)),(int(centroide[0]-200),int(centroide[1])+200), self.colorAzul,1)

            cx = centroide[0]
            cy = centroide[1]
            listaPuntos = []
            polyShape = ""
            contornos = contornos[0]

            # construir poligono shapely
            puntoInicial = contornos[0][0]
            for i, puntoContorno in enumerate(contornos):
                x = puntoContorno[0][0]  # x
                y = puntoContorno[0][1]  # y
                listaPuntos.append((x,y))
                if i < len(contornos)-1:
                    polyShape = polyShape + str(x) + " " + str(y)
                    polyShape = polyShape + ", "
                else:
                    polyShape = polyShape + str(puntoInicial[0]) + " " + str(puntoInicial[1])
            polygon = "POLYGON ((" + polyShape + "))"
            #print(polygon)
            poly = loads(polygon)

            pts = np.array(listaPuntos, np.int32)
            #cv2.polylines(self.imagen, [pts], True, self.colorAzul, 2)

            # construir line shapely
            lineShape1 = "("+str((centroide[0])-200) + " " + str((centroide[1]))+ ", " +str((centroide[0])+200) + " " + str((centroide[1]))+")"
            lineShape2 = "("+str((centroide[0])) + " " + str((centroide[1]) - 200) + ", " + str((centroide[0])) + " " + str((centroide[1]) + 200)+")"
            lineShape3 = "("+str((centroide[0]) - 200) + " " + str((centroide[1]) - 200) + ", " + str((centroide[0]) + 200) + " " + str((centroide[1]) + 200)+")"
            lineShape4 = "("+str((centroide[0]) + 200) + " " + str((centroide[1]) - 200) + ", " + str((centroide[0]) - 200) + " " + str((centroide[1]) + 200)+")"
            lineString = "MULTILINESTRING (" + lineShape1 + ", " + lineShape2 + ", " + lineShape3 + ", " + lineShape4 + ")"
            #print(lineString)
            line = loads(lineString)
            self.calcularIntersection(poly,line,frame, centroide, imagenID)

            cv2.imwrite("imagen.png", self.imagen)
            #img = cv2.resize(self.imagen, (800, 600))
            #cv2.imshow("fuego", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            print("error al cargar la imagen")

        return (frame, self.segmentos)

    def calcularIntersection(self, poly, line, frame, centroide, imagenID):
        segmentosImagen = []
        intersection = poly.exterior.intersection(line)
        if intersection.is_empty:
            print("shape don't intersect")
        elif intersection.geom_type.startswith('Multi') or intersection.geom_type == 'GeometryCollection':
            for i, shp in enumerate(intersection):
                if isinstance(shp, Point):
                    cv2.circle(frame, (int(shp.x), int(shp.y)), 2, self.colorAmarillo, 2)
                    #cv2.putText(frame,str(i),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colorAzul, 2, cv2.LINE_AA)
                    distancia = self.calcularDistanciaEntreDosPuntos(centroide[0], centroide[1], shp.x, shp.y)
                    segmento = Segmento(imagenID, i, centroide, (shp.x,shp.y), distancia)
                    segmentosImagen.append(segmento)
                else:
                    x,y = shp.xy
                    cv2.circle(frame, (int(x[0]), int(y[0])), 2, self.colorAmarillo, 2)
                    distancia = self.calcularDistanciaEntreDosPuntos(centroide[0], centroide[1], x[0], y[0])
                    segmento = Segmento(imagenID, i, centroide, (x[0], y[0]), distancia)
                    segmentosImagen.append(segmento)
            self.segmentos.append(segmentosImagen)
        else:
            print(intersection)

    def aplicarDilate(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        imageDilate = cv2.dilate(image, kernel)
        return imageDilate

    def segmentarRojoAmarrillo(self, imagenHSV):
        rojoLowerDown = np.array([0, 100, 100])
        rojoLowerUp = np.array([10, 255, 255])
        rojoUpperDown = np.array([160, 100, 100])
        rojoUpperUp = np.array([179, 255, 255])
        #yellowLower = np.array([22, 60, 200])
        #yellowLower = np.array([11, 60, 200])
        #yellowUpper = np.array([60, 255, 255])
        #yellowUpper = np.array([32, 255, 255])
        mascara1 = cv2.inRange(imagenHSV, rojoLowerDown, rojoLowerUp)
        mascara2 = cv2.inRange(imagenHSV, rojoUpperDown, rojoUpperUp)
        # maskImage = cv2.bitwise_or(mascara1, mascara2)
        mascaraRojo = cv2.addWeighted(mascara1, 1.0, mascara2, 1.0, 0.0)
        imagenRojo = cv2.bitwise_and(self.imagen, self.imagen, mascaraRojo);
        #cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentacion/filtros/imagenRoja2.png", mascaraRoja)
        #mascaraAmarilla = cv2.inRange(imagenHSV, yellowLower, yellowUpper)
        #mascaraImage = cv2.addWeighted(mascaraRoja, 1.0, mascaraAmarilla, 1.0, 0.0)
        imask = mascaraRojo > 0

        #green = np.zeros_like(self.imagen, np.uint8)
        #green[imask] = self.imagen[imask]
        #cv2.imwrite("Rojo.png", green)

        return mascaraRojo

    def segmentarVerde(self, imagenHSV):
        greenLowerDown = np.array([36, 25, 25])
        greenUpperUp = np.array([70, 255, 255])
        mascara = cv2.inRange(imagenHSV, greenLowerDown, greenUpperUp)
        imask = mascara>0

        #green = np.zeros_like(self.imagen, np.uint8)
        #green[imask] = self.imagen[imask]
        #cv2.imwrite("Verde.png",green)

        # maskImage = cv2.bitwise_or(mascara1, mascara2)
        #mascaraRoja = cv2.addWeighted(mascara1, 1.0, mascara2, 1.0, 0.0)
        # imagenRoja = cv2.bitwise_and(frame, frame, mascaraRoja);
        mascaraImage = cv2.addWeighted(mascara, 1.0, mascara, 1.0, 0.0)
        return mascaraImage

    def encontrarMomentosHu(self, contorno):
        momentos = cv2.moments(contorno)
        huMomentos = cv2.HuMoments(momentos)
        return huMomentos

    def encontrarCentroide(self, contorno):
        momentos = cv2.moments(contorno)
        huMomentos = cv2.HuMoments(momentos)
        x = 0
        y = 0
        if momentos['m00'] != 0:
            x = int(momentos['m10']/momentos['m00'])
            y = int(momentos['m01']/momentos['m00'])
        return (x,y)

    def calcularDistanciaEntreDosPuntos(self, puntoX1, puntoY1, puntoX2, puntoY2):
        distancia = math.sqrt((puntoX2-puntoX1)**2 + (puntoY2-puntoY1)**2)
        return distancia