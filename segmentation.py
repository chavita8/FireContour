import cv2
import numpy as np
import math
from shapely.wkt import loads
from shapely.geometry.point import Point
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, MultiLineString
from segmento import Segmento
from contour import Contour
import random
from ray import Ray

class Segmentation(object):
    def __init__(self):
        self.image = cv2.imread("fire.png")
        self.blueColor = (255, 0, 0)
        self.greenColor = (0, 255, 0)
        self.redColor = (0, 0, 255)
        self.yellowColor = (0, 255, 255)
        self.pinkColor = (255, 255, 0)
        self.nColor = (255, 0, 255)
        self.centroidList = []
        self.invariantsHuMoments = []
        self.invariantsSiftPoints = []
        self.invariantsSurtPoints = []
        self.segments = []

    def segmentImage(self, numberRays, numberContours):
        if np.any(self.image):
            imageNoise = cv2.medianBlur(self.image, 11)
            imageHSV = cv2.cvtColor(imageNoise, cv2.COLOR_BGR2HSV)
            channelHSV = cv2.split(imageHSV)
            (channelH, channelS, channelV) = channelHSV
            cannyImage = cv2.Canny(channelV, 127, 255)
            cv2.imwrite("Canny.png", cannyImage)
            imageRedYellow = self.segmentRedYellow(imageHSV)
            cv2.imwrite("RedMask.png", imageRedYellow)
            greenImage = self.segmentGreen(imageHSV)
            cv2.imwrite("GreenMask.png", greenImage)
            cannyImageRed = cv2.Canny(imageRedYellow, 127, 255)
            cv2.imwrite("RedCanny.png", cannyImageRed)
            _,contours,_ = cv2.findContours(cannyImageRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            centroid = self.findCentroid(contours[0])
            cv2.circle(self.image, centroid, 3, self.pinkColor, 3)

            polygons, res = self.generatePolygons(numberContours,contours)
            lastContour = []
            lastContour.append(res)

            raysList = self.generateRays(centroid, lastContour, numberRays)
            intersections = self.intersectPolygons(polygons,raysList)
            self.drawPoints(intersections)
            print("Puntos")
            dictRays = self.showRays(raysList)
            distanceDiff = self.calcularDiferenciaDistancia(dictRays,0)
            print(distanceDiff)
            #speeds = self.calcularVelocidadRayos(numberRays)
            #variations = self.calcularVariacionDistanciaRayos(numberRays)

            cv2.imwrite("image.png", self.image)
            cv2.imshow("image",self.image)
            cv2.waitKey(0)
        else:
            print("error loading image")

    def showRays(self, raysList):
        dict = {}
        for ray in raysList:
            intersectionList = ray.intersectionList
            intersections = []
            for intersection in intersectionList:
                intersections.append(intersection)
            dict[ray.rayId] = intersections
        print(dict)
        return dict

    def calcularDiferenciaDistancia(self, rayList, rayID):
        variacionDistancia = []
        intersectionRays = rayList[rayID]
        print(intersectionRays)
        i = 0
        while i < len(intersectionRays):
            if i + 1 < len(intersectionRays):
                intersection1 = intersectionRays[i]
                intersection2 = intersectionRays[i + 1]
                distancia1 = intersection1.distance
                distancia2 = intersection2.distance
                variacion = distancia2 - distancia1
                variacionDistancia.append(variacion)
            i = i + 1
        return variacionDistancia

    def generatePolygons(self, numC, contour):
        listPolygonsShapely = []
        res, contoursList = self.generateContours(numC,contour[0],[])
        largestContour = self.largestContour(contoursList)
        #cv2.drawContours(self.image, lists, -1, self.blueColor, 3)
        for contourObj in contoursList:
            cnt = contourObj.contour
            color = contourObj.color
            cv2.drawContours(self.image, cnt, -1, color, 3)
            polygon = []
            array = cnt
            startingPoint = (array[0][0][0], array[0][0][1])
            for pointArray in array:
                tupla = (pointArray[0][0], pointArray[0][1])
                polygon.append(tupla)
            polygon.append(startingPoint)
            poligonoShapely = Polygon(polygon)
            listPolygonsShapely.append(poligonoShapely)
        multiPolygon = MultiPolygon(listPolygonsShapely)
        return (multiPolygon, largestContour.contour)

    def largestContour(self,contoursList):
        largest = None
        maximunArea = -1
        for contour in contoursList:
            area = contour.contourArea()
            if area > maximunArea:
                largest = contour
                maximunArea = area
        return largest

    def generateContours(self, numC, contour, list):
        if numC == 1:
            lastContour = Contour(contour, self.blueColor, "first")
            list.append(lastContour)
        else:
            lastContour, list = self.generateContours(numC-1,contour,list)
            scaleIncrease = random.uniform(1.1, 1.6)
            scaleDecrease = random.uniform(1.1, 1.3)
            operator = random.randint(0,3)
            print("OPERATOR")
            print(operator)
            if operator == 1:
                newSize = self.scaleContour(lastContour.contour,scaleDecrease,operator)
                contourObject = Contour(newSize, self.nColor, "decrease")
                print("SCALE DECREASE")
                print(scaleDecrease)
            else:
                newSize = self.scaleContour(lastContour.contour, scaleIncrease)
                contourObject = Contour(newSize, self.blueColor, "increase")
                print("SCALE INCREASE")
                print(scaleIncrease)
            #lastContour = newSize
            lastContour = contourObject
            list.append(lastContour)
        return (lastContour,list)

    def scaleContour(self, contour, scale, decrease=None):
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        contourNorm = contour - [cx, cy]
        if decrease == None:
            contourScaled = contourNorm * scale
        else:
            contourScaled = contourNorm / scale
        contourScaled = contourScaled + [cx, cy]
        contourScaled = contourScaled.astype(np.int32)
        return contourScaled

    def generateRays(self,centroid,contours,numL):
        coords = []
        north = self.north(centroid,contours)
        sub = self.sub(centroid,contours)
        east = self.east(centroid,contours)
        west = self.west(centroid,contours)
        northeast = self.puntoMedio(north,east)
        subeast = self.puntoMedio(sub,east)
        subwest = self.puntoMedio(sub,west)
        northwest = self.puntoMedio(north,west)
        coords.append(north)
        coords.append(northeast)
        coords.append(east)
        coords.append(subeast)
        coords.append(sub)
        coords.append(subwest)
        coords.append(west)
        coords.append(northwest)
        self.writeImage("N", north[0],north[1])
        self.writeImage("S", sub[0],sub[1])
        self.writeImage("E", east[0],east[1])
        self.writeImage("O", west[0],west[1])
        if numL <= 8:
            pointsDirection = coords[:numL]
        else:
            pointsDirection = self.calculateMidlepoints(coords, numL-len(coords))
        self.traceRays(self.image, centroid, pointsDirection)
        rays = self.createRays(centroid,pointsDirection)
        #multiline = self.buildMultilineShapely(centroid,pointsDirection)
        return rays

    def buildMultilineShapely(self, centroid, spaceWork):
        lineString = "MULTILINESTRING ("
        for i,punto in enumerate(spaceWork):
            lineShape = "("+str(centroid[0])+" "+str(centroid[1])+", "+str(punto[0])+" "+str(punto[1])+")"
            if i < len(spaceWork) - 1:
                lineString = lineString + lineShape +", "
            else:
                lineString = lineString + lineShape + ")"
        line = loads(lineString)
        return line

    def createRays(self, centroid, pointsDirection):
        rays = []
        pointCentroid = Point(centroid)
        for i, point in enumerate (pointsDirection):
            ray = Ray(i, pointCentroid, point)
            rays.append(ray)
        return rays

    def drawPoints(self,intersections):
        for list in intersections:
            for intersection in list:
                point = intersection.intersectionPoint
                if isinstance(point, Point):
                    x = point.x
                    y = point.y
                    cv2.circle(self.image, (int(x), int(y)), 2, self.yellowColor, 2)

    def intersectPolygons(self,polygons,rays):
        listIntersections = []
        for i,polygon in enumerate(polygons):
            for ray in rays:
                intersection = ray.intersect(polygon,i)
                listIntersections.append(intersection)
        return listIntersections

    def calculateMidlepoints(self, listaCoords, num):
        counter = num
        i = 0 
        while counter > 0: 
            if i+1 < len(listaCoords):
                p1 = listaCoords[i]
                p2 = listaCoords[i+1]
                pM = self.middlePoint(p1,p2)
                listaCoords.insert(i+1,pM)
                i += 2
            else:
                p1 = listaCoords[i]
                p2 = listaCoords[0]
                pM = self.middlePoint(p1,p2)
                listaCoords.append(pM)
                i = 0
            counter -= 1
        return listaCoords

    def middlePoint(self, coordA, coordB):
        xA = coordA[0]
        yA = coordA[1]
        xB = coordB[0]
        yB = coordB[1]
        xC = (xA+xB)/2
        yC = (yA+yB)/2
        res = (int(xC),int(yC))
        return res

    def puntoMedio(self, coord1, coord2):
        yCoord1 = coord1[1]
        yCoord2 = coord2[1]
        diff = (yCoord2 - yCoord1) / 2
        res = (coord2[0], int(yCoord1 + diff))
        return res

    def traceRays(self, img, centroid, rays):
        for point in rays:
            cv2.circle(img,point,1,self.pinkColor,1)
            cv2.line(img, (centroid[0], centroid[1]),(point[0],point[1]), self.pinkColor,1)

    def north(self,centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extTop = list(c[c[:, :, 1].argmin()][0])
        extTop[0] = centroid[0]
        res = tuple(extTop)
        return res

    def sub(self,centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extBot = list(c[c[:, :, 1].argmax()][0])
        extBot[0] = centroid[0]
        res = tuple(extBot)
        return res

    def east(self,centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extRight = list(c[c[:, :, 0].argmax()][0])
        extRight[1] = centroid[1]
        res = tuple(extRight)
        return res

    def west(self,centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extLeft = list(c[c[:, :, 0].argmin()][0])
        extLeft[1] = centroid[1]
        res = tuple(extLeft)
        return res

    def calculateIntersection(self, contornoID, polygon, lines, centroid):
        intersection = polygon.exterior.intersection(lines)
        if intersection.is_empty:
            print("shape don't intersect")
        elif intersection.geom_type.startswith('Multi') or intersection.geom_type == 'GeometryCollection':
            for i, shp in enumerate(intersection):
                if isinstance(shp, Point):
                    cv2.circle(self.image, (int(shp.x), int(shp.y)), 2, self.yellowColor, 2)
                    distance = self.calculateDistanceBetweenTwoPoints(centroid[0], centroid[1], shp.x, shp.y)
                    segment = Segmento(contornoID, i, centroid, (shp.x,shp.y), distance)
                    #self.writeImage(str(i),int(shp.x),int(shp.y))
                    self.segments.append(segment)
                else:
                    x,y = shp.xy
                    cv2.circle(self.image, (int(x[0]), int(y[0])), 2, self.yellowColor, 2)
                    distance = self.calculateDistanceBetweenTwoPoints(centroid[0], centroid[1], x[0], y[0])
                    segment = Segmento(contornoID,i, centroid, (x[0], y[0]), distance)
                    #self.writeImage(str(i), int(x[0]), int(y[0]))
                    self.segments.append(segment)
        else:
            print(intersection)

    def calculateSpeed(self, segmentsList):
        speedList = []
        tamano = len(segmentsList)
        i = 0
        while i + 1 < tamano:
            segment1 = segmentsList[i]
            segment2 = segmentsList[i + 1]
            d1 = segment1.distancia
            d2 = segment2.distancia
            v = 0
            dX = d2 - d1
            dT = segment1.contornoID + 1
            if dT != 0:
                v = dX / dT
            speedList.append(v)
            i += 1
        return speedList

    def calculateDistanceBetweenTwoPoints(self, puntoX1, puntoY1, puntoX2, puntoY2):
        distancia = math.sqrt((puntoX2-puntoX1)**2 + (puntoY2-puntoY1)**2)
        return distancia

    def segmentRedYellow(self, imageHSV):
        rojoLowerDown = np.array([0, 100, 100])
        rojoLowerUp = np.array([10, 255, 255])
        rojoUpperDown = np.array([160, 100, 100])
        rojoUpperUp = np.array([179, 255, 255])
        mascara1 = cv2.inRange(imageHSV, rojoLowerDown, rojoLowerUp)
        mascara2 = cv2.inRange(imageHSV, rojoUpperDown, rojoUpperUp)
        # maskImage = cv2.bitwise_or(mascara1, mascara2)
        mascaraRojo = cv2.addWeighted(mascara1, 1.0, mascara2, 1.0, 0.0)
        imagenRojo = cv2.bitwise_and(self.image, self.image, mascaraRojo);
        #cv2.imwrite("/home/chavita/Documentos/tesis/codigos/Segmentacion/filtros/imagenRoja2.png", mascaraRoja)
        #mascaraAmarilla = cv2.inRange(imagenHSV, yellowLower, yellowUpper)
        #mascaraImage = cv2.addWeighted(mascaraRoja, 1.0, mascaraAmarilla, 1.0, 0.0)
        imask = mascaraRojo > 0
        #green = np.zeros_like(self.imagen, np.uint8)
        #green[imask] = self.imagen[imask]
        #cv2.imwrite("Rojo.png", green)
        return mascaraRojo

    def segmentGreen(self, imagenHSV):
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

    def findCentroid(self, contorno):
        momentos = cv2.moments(contorno)
        huMomentos = cv2.HuMoments(momentos)
        x = 0
        y = 0
        if momentos['m00'] != 0:
            x = int(momentos['m10']/momentos['m00'])
            y = int(momentos['m01']/momentos['m00'])
        return (x,y)

    def obtenerSegmentosPorContornoID(self, contornoID):
        segmentosContorno = []
        for segmento in self.segmentos:
            if segmento.contornoID == contornoID:
                segmentosContorno.append(segmento)
        return segmentosContorno

    def obtenerSegmentosPorID(self, segmentoID):
        segmentosContorno = []
        for segmento in self.segments:
            if segmento.segmentoID == segmentoID:
                segmentosContorno.append(segmento)
        return segmentosContorno

    def obtenerSegmentosPorSegmentosID(self, segmentoID):
        segmentosContorno = []
        for segmento in self.segments:
            if segmento.segmentoID == segmentoID:
                segmentosContorno.append(segmento)
        return segmentosContorno

    def calcularVelocidadRayos(self,numeroRayos):
        # Calcular velocidad de todos los rayos trazados
        count = 0
        velocidadRayos = {}
        while count < numeroRayos:
            segmentos = self.obtenerSegmentosPorSegmentosID(count)
            velocidad = self.calculateSpeed(segmentos)
            velocidadRayos[count] = velocidad
            count = count + 1
        print("VelocidadRayos: " + str(velocidadRayos))
        return velocidadRayos

    def calcularVariacionDistanciaRayos(self,numeroRayos):
        # Calcular variacion de distancia de todos los rayos trazados
        count = 0
        distanciaRayos = {}
        while count < numeroRayos:
            segmentos = self.obtenerSegmentosPorSegmentosID(count)
            distancia = self.calcularDiferenciaDistancia(count)
            distanciaRayos[count] = distancia
            count = count + 1
        print("Variacion Distancia Rayos: " + str(distanciaRayos))
        return distanciaRayos

    def writeImage(self, palabra, x, y):
        fontFace = cv2.FONT_ITALIC
        fontScale = 0.5
        espesor = 1
        cv2.putText(self.image, palabra, (x,y),fontFace, fontScale, self.pinkColor, espesor)