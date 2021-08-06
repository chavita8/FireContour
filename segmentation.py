import cv2
import numpy as np
import math
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, MultiLineString
from contour import Contour
import random
import time
import csv
import pandas as pd
import imutils

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
            kernelmatrix = np.ones((5, 5), np.uint8)
            resultimage = cv2.dilate(cannyImageRed, kernelmatrix)
            cv2.imwrite("dilate.png", resultimage)
            #cannyDilate = cv2.dilate(cannyImageRed,3,iterations=5)
            _,contours,_ = cv2.findContours(resultimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #print("Scale")
            M = cv2.moments(contours[0])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #print(cx)
            #print(cy)
            contourNorm = contours[0] - [cx, cy]
            #print(len(contourNorm))
            #print(contourNorm)
            contourScaled = contourNorm * 1.01
            #print(len(contourScaled))
            #print(contourScaled)

            centroid = self.findCentroid(contours[0])
            cv2.circle(self.image, centroid, 3, self.whiteColor, 3)

            polygons, res = self.simulateContours(contours[0],numberContours, centroid)
            print("POLYGONS : " + str(len(polygons)))
            lastContour = []
            lastContour.append(res)
            raysList = self.generateRays(centroid, lastContour, numberRays)
            intersections = self.intersectBetweenRaysAndPolygon(polygons,raysList)
            rayId = 5
            ray = raysList[rayId]
            distances = ray.obtenerDistances()
            print("DISTANCIAS:"+ str(len(distances)))
            print(distances)
            self.generateCSV(distances, rayId)
            plt.plot(distances)
            plt.show()
            self.drawPoints(intersections)
            self.drawRayId(ray, rayId)
            self.resizeImage(numberContours)
            cv2.imwrite("image.png", self.image)
            cv2.imshow("image",self.image)
            cv2.waitKey(0)
        else:
            print("error loading image")

    def simulateContours(self,contour, numberContours, centroid):
        contours_list = []

        def exp(x):
            return np.exp(x)
        x = np.linspace(0,1.1,numberContours)
        y = exp(x)
        firstContour = Contour(contour, self.blueColor, "first")
        contours_list.append(firstContour)
        img = np.zeros((153, 203, 3), np.uint8)
        cv2.drawContours(img,firstContour.contour,-1,self.whiteColor,3)
        cv2.imwrite("newImage.png",img)

        for i,value in enumerate(x):
            scale = 1.1
            #scale = random.uniform(1.01, 1.02)
            #scale = 1.1 #max value
            list_size = len(contours_list)
            last_contour = contours_list[list_size - 1]
            print("crece")
            new_contour = self.scaleContour(last_contour.contour, scale)
            scaled_contour = Contour(new_contour, self.blueColor, "increace")
            contours_list.append(scaled_contour)

        return self.generatePolygons(contours_list)

    def scaleContour(self, contour, scale, decrease=None):
            M = cv2.moments(contour)
            contourScaled = contour
            if M['m00'] != 0:
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

    def intersectBetweenRaysAndPolygon(self,polygons,rays):
        listIntersections = []
        for i,polygon in enumerate(polygons):
            for ray in rays:
                intersection = ray.intersect(polygon,i)
                listIntersections.append(intersection)
        return listIntersections

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

    def generateCSV(self,distances, rayID):
        #times = np.array(range(0, len(distances)))
        times = np.linspace(0.1, 5.6, len(distances))
        X = times.reshape(-1, 1)
        Y = np.array(distances).reshape(-1,1)
        print("Distancias csv: " + str(len(distances)))
        csv_arr = []
        csv_arr.append(["tiempo","distancia"])
        for i,value in enumerate(X):
            arr = []
            x = X[i]
            y = Y[i]
            arr.append(x[0])
            arr.append(y[0])
            csv_arr.append(arr)
        filename = 'distanciasRayo'+str(rayID)+'.csv'
        myFile = open(filename, 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(csv_arr)
        #df = pd.read_csv(filename)
        #print("DataFrame")
        #print(df)

    def drawRayId(self,ray, rayId):
        x = 30
        y = 550
        intersections = ray.intersectionsList
        points = []
        for intersection in intersections:
            point = intersection.intersectionPoint
            tuple = (int(point.x), int(point.y))
            points.append(tuple)
        word1 = "Ray:  " + str(rayId)
        #print(word1)
        self.writeImageText(word1, x, y, self.whiteColor)
        word7 = str(points)
        #print(word7)
        self.writeImageText(word7, x, y + 10, self.blackColor)
        word2 ="Diferencia Distances"
        #print(word2)
        self.writeImageText(word2, x, y+35, self.whiteColor)
        distanceDiff = self.calcularDiferenciaDistancia(intersections)
        word3 = str(distanceDiff)
        #print(word3)
        self.writeImageText(word3, x, y+50, self.blackColor)
        word4 = "speeds"
        #print(word4)
        self.writeImageText(word4, x, y+75, self.blackColor)
        speeds = self.calculateSpeed(intersections)
        word5 = str(speeds)
        #print(word5)
        self.writeImageText(word5, x, y+90, self.blackColor)

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
        yCoord1 = np.array(coord1[1],dtype='int64')
        yCoord2 = np.array(coord2[1],dtype='int64')
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

    def resizeImage(self,resizeNumber):
        img = cv2.imread("imagenPeque.png")
        height = img.shape[0]
        width = img.shape[1]
        for i in range(resizeNumber):
            resized = imutils.resize(img,height=height+10,width=width+50,inter=cv2.INTER_AREA)
            height = resized.shape[0]
            width = resized.shape[1]
            cv2.imwrite("imagen"+str(i)+".png",resized)