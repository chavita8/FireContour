import cv2
import numpy as np
import math
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, MultiLineString
from sklearn.linear_model import LinearRegression
from contour import Contour
import random
from ray import Ray

class Segmentation(object):
    def __init__(self):
        self.image = cv2.imread("fire.png")
        self.whiteColor = (255, 255, 255)
        self.blackColor = (0, 0, 0)
        self.blueColor = (255, 0, 0)
        self.greenColor = (0, 255, 0)
        self.redColor = (0, 0, 255)
        self.yellowColor = (0, 255, 255)
        self.cianColor = (255, 255, 0)
        self.pinkColor = (255, 0, 255)
        self.invariantsHuMoments = []
        self.invariantsSiftPoints = []
        self.invariantsSurtPoints = []

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
            contours,_ = cv2.findContours(cannyImageRed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            centroid = self.findCentroid(contours[0])
            cv2.circle(self.image, centroid, 3, self.whiteColor, 3)

            polygons, res = self.simulateContours(contours[0],numberContours, centroid)
            print("POLYGONS : " + str(len(polygons)))
            lastContour = []
            lastContour.append(res)
            raysList = self.generateRays(centroid, lastContour, numberRays)
            intersections = self.intersectBetweenRaysAndPolygon(polygons,raysList)

            self.drawPoints(intersections)
            self.drawRayId(raysList,5)
            self.mostrarGraficoDistancia(raysList,5)
            #self.linearRegression(raysList)
            cv2.imwrite("image.png", self.image)
            cv2.imshow("image",self.image)
            cv2.waitKey(0)
        else:
            print("error loading image")

    def simulateContours(self,contour, numberContours, centroid):
        contours_list = []
        distances = []
        """
        N = numberContours  # number of data points
        times = np.linspace(0, 2 * np.pi, N)
        f = 1.15247  # Optional!! Advised not to use
        data = 3.0 * np.sin(f * times + 0.001) + 0.5 + np.random.randn(N)  # create artificial data with noise
        guess_mean = np.mean(data)
        guess_std = 3 * np.std(data) / (2 ** 0.5) / (2 ** 0.5)
        guess_phase = 0
        oscillationSine = guess_std * np.sin(times + guess_phase) + guess_mean
        """
        def sine(x):
            equation = -15*(np.cos(((math.pi/14)*x) - ((3*math.pi)/14))+17)
            return equation
        times = []
        for time in range(0,200):
            times.append(time)
        times = np.array(times)
        oscillationSine = sine(times)
        plt.scatter(times,oscillationSine)
        plt.show()
        print("Oscillation Sine")
        print(oscillationSine)
        firstContour = Contour(contour, self.blueColor, "first")
        contours_list.append(firstContour)
        maxValue = -100.0

        for i,value in enumerate(oscillationSine):
            if i+1 < len(oscillationSine):
                sinusoidePoint = (times[i],oscillationSine[i])
                distance = self.calculateDistanceBetweenTwoPoints(centroid[0], centroid[1], sinusoidePoint[0], sinusoidePoint[1])
                distances.append(distance)
                scale = 1.3
                #scale = random.uniform(1.1, 1.4)
                list_size = len(contours_list)
                last_contour = contours_list[list_size - 1]
                scaled_contour = None;
                #print("value: " + str(value))
                #print("maxValue: " + str(maxValue))
                if value > maxValue:
                    print("crece")
                    new_contour = self.scaleContour(last_contour.contour, scale)
                    scaled_contour = Contour(new_contour, self.blueColor, "increace")
                else:
                    if value < maxValue:
                        print("decrece")
                        new_contour = self.scaleContour(last_contour.contour, scale, 1)
                        scaled_contour = Contour(new_contour, self.pinkColor, "decreace")
                    else:
                        print("Se mantiene ")
                if scaled_contour != None:
                    contours_list.append(scaled_contour)
                maxValue = value
        #print("lISTA DIST:"+ str(distances))
        #plt.plot(distances)
        #plt.scatter(times, oscillationSine)
        #plt.show()
        #self.linearRegression(times,distances)
        return self.generatePolygons(contours_list)

    def generatePolygons(self, contours_list):
        listPolygonsShapely = []
        largestContour = self.largestContour(contours_list)

        for contourObj in contours_list:
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

    def generateRays(self, centroid, lastContour, numL):
        coords = []
        north = self.north(centroid, lastContour)
        sub = self.sub(centroid, lastContour)
        east = self.east(centroid, lastContour)
        west = self.west(centroid, lastContour)
        northeast = self.puntoMedio(north, east)
        subeast = self.puntoMedio(sub, east)
        subwest = self.puntoMedio(sub, west)
        northwest = self.puntoMedio(north, west)
        coords.append(north)
        coords.append(northeast)
        coords.append(east)
        coords.append(subeast)
        coords.append(sub)
        coords.append(subwest)
        coords.append(west)
        coords.append(northwest)
        # self.writeImage("N", north[0],north[1], self.whiteColor)
        # self.writeImage("S", sub[0],sub[1], self.whiteColor)
        # self.writeImage("E", east[0],east[1], self.whiteColor)
        # self.writeImage("O", west[0],west[1], self.whiteColor)
        if numL <= 8:
            pointsDirection = coords[:numL]
        else:
            pointsDirection = self.calculateMidlepoints(coords, numL - len(coords))
        self.drawRays(self.image, centroid, pointsDirection)
        rays = self.createRays(centroid, pointsDirection)
        return rays

    def createRays(self, centroid, pointsDirection):
        rays = []
        pointCentroid = Point(centroid)
        for i, point in enumerate(pointsDirection):
            ray = Ray(i, pointCentroid, point)
            rays.append(ray)
            self.writeImage(str(i), int(point[0]), int(point[1]), self.blackColor)
        return rays

    def intersectBetweenRaysAndPolygon(self,polygons,rays):
        listIntersections = []
        for i,polygon in enumerate(polygons):
            for ray in rays:
                intersection = ray.intersect(polygon,i)
                listIntersections.append(intersection)
        return listIntersections

    def drawRayId(self,raysList, rayId):
        x = 30
        y = 550
        ray = raysList[rayId]
        intersections = ray.intersectionsList
        points = []
        for intersection in intersections:
            point = intersection.intersectionPoint
            tuple = (int(point.x), int(point.y))
            points.append(tuple)
        word1 = "Ray:  " + str(rayId)
        print(word1)
        self.writeImageText(word1, x, y, self.whiteColor)
        word7 = str(points)
        print(word7)
        self.writeImageText(word7, x, y + 10, self.blackColor)
        word2 ="Diferencia Distances"
        print(word2)
        self.writeImageText(word2, x, y+35, self.whiteColor)
        distanceDiff = self.calcularDiferenciaDistancia(intersections)
        word3 = str(distanceDiff)
        print(word3)
        self.writeImageText(word3, x, y+50, self.blackColor)
        word4 = "speeds"
        print(word4)
        self.writeImageText(word4, x, y+75, self.blackColor)
        speeds = self.calculateSpeed(intersections)
        word5 = str(speeds)
        print(word5)
        self.writeImageText(word5, x, y+90, self.blackColor)

    def mostrarGraficoDistancia(self,rayList,rayID):
        ray = rayList[rayID]
        distances = ray.obtenerDistances();
        print("DISTANCES")
        print(distances)
        plt.plot(distances)
        plt.show()

    def calcularDiferenciaDistancia(self, intersections):
        variacionDistancia = []
        i = 0
        while i < len(intersections):
            if i + 1 < len(intersections):
                intersection1 = intersections[i]
                intersection2 = intersections[i + 1]
                distancia1 = intersection1.distance
                distancia2 = intersection2.distance
                variacion = distancia2 - distancia1
                variacionDistancia.append(variacion)
            i = i + 1
        return variacionDistancia

    def calculateSpeed(self, intersections):
        speedList = []
        i = 0
        while i + 1 < len(intersections):
            intersection1 = intersections[i]
            intersection2 = intersections[i + 1]
            distancia1 = intersection1.distance
            distancia2 = intersection2.distance
            v = 0
            dX = abs(distancia2 - distancia1)
            dT = intersection1.contourId + 1
            if dT != 0:
                v = dX / dT
            speedList.append(v)
            i += 1
        return speedList

    def linearRegression(self,distances):
        """! Metodo que entrena un algoritmo de regresion lineal
        @param times Lista de valores en x
        @param distances Lista de Valores en y
        """
        regresion_lineal = LinearRegression()
        # instruimos a la regresion lineal que aprenda de los datos (x,y)
        #x = np.arange(0,len(distances),1)
        regresion_lineal.fit(times.reshape(-1, 1),distances)

        # vemos los parametros que ha estimado la regresion lineal
        w = regresion_lineal.coef_
        b = regresion_lineal.intercept_

        print('w = ' + str(w))
        print('b = ' + str(b))

        # vamos a predecir y = regresion_lineal(5)
        #nuevo_x = np.array([0])
        prediccion = regresion_lineal.predict(times.reshape(-1, 1))
        plt.scatter(times,prediccion)
        print(prediccion)

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

    def drawPoints(self,intersections):
        for list in intersections:
            for intersection in list:
                point = intersection.intersectionPoint
                if isinstance(point, Point):
                    x = point.x
                    y = point.y
                    cv2.circle(self.image, (int(x), int(y)), 2, self.yellowColor, 2)
                    self.writeImage(str(intersection.contourId),int(x),int(y), self.cianColor)

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

    def drawRays(self, img, centroid, rays):
        for point in rays:
            cv2.circle(img,point,1,self.whiteColor,1)
            cv2.line(img, (centroid[0], centroid[1]),(point[0],point[1]), self.whiteColor,1)

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

    def writeImage(self, palabra, x, y, color):
        fontFace = cv2.FONT_ITALIC
        fontScale = 0.3
        espesor = 0
        cv2.putText(self.image, palabra, (x,y),fontFace, fontScale, color, espesor)

    def writeImageText(self, palabra, x, y, color):
        fontFace = cv2.FONT_ITALIC
        fontScale = 0.4
        espesor = 1
        cv2.putText(self.image, palabra, (x,y),fontFace, fontScale, color, espesor)

