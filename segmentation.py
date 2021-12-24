import cv2
import numpy as np
import math
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, MultiLineString, LineString, LinearRing, GeometryCollection
from contour import Contour
import random
import imutils
from shapely.validation import explain_validity
from shapely.validation import make_valid
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos
from ray import Ray

class segmentation(object):
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
        self.grayColor = (128, 128, 128)
        self.centroidList = []
        self.pointList = []

    def segmentImage(self, numberRays, numberContours):
        if np.any(self.image):
            imageNoise = cv2.medianBlur(self.image, 11)
            imageHSV = cv2.cvtColor(imageNoise, cv2.COLOR_BGR2HSV)
            channelHSV = cv2.split(imageHSV)
            (channelH, channelS, channelV) = channelHSV
            cannyImage = cv2.Canny(channelV, 127, 255)
            imageRedYellow = self.segmentRedYellow(imageHSV)
            cannyImageRed = cv2.Canny(imageRedYellow, 127, 255)
            kernelmatrix = np.ones((5, 5), np.uint8)
            resultimage = cv2.dilate(cannyImageRed, kernelmatrix)
            cv2.imwrite("dilate.png", resultimage)
            _, contours, _ = cv2.findContours(resultimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            centroidIni = self.findCentroid(contours[0])
            cv2.circle(self.image, centroidIni, 3, self.whiteColor, 3)
            print("Centroide Inicial: " + str(centroidIni))

            # f(x) = ax^2 + bx + c
            # f(x) = -2x ^ 2 + 3x + 0
            listVarY = self.deformFuncionParabola(-2, 3, 0, numberContours)
            print(listVarY)

            alphaL = 30
            alphaR = 0
            largestContour = None
            maximunArea = -1
            multiPolygonList = []

            limite = numberContours / 3
            contador = 0
            for c in range(numberContours):
                randR = random.randint(60, 120)
                alphaR = alphaR + randR
                randL = random.randint(150, 280)
                alphaL = alphaL + randL
                contour1 = self.resizeContours(resultimage, centroidIni, alphaR, 0)
                contour2 = self.resizeContours(resultimage, centroidIni, alphaL, 0)
                if numberContours > 5:
                    if contador >= 0 and contador <= limite:
                        contourDeform = self.deformContours(contour1, contour2, centroidIni, 0)
                    if contador > limite and contador <= limite * 2:
                        contourDeform = self.deformContours(contour1, contour2, centroidIni, 1)
                    if contador > limite * 2 and contador <= limite * 3:
                        contourDeform = self.deformContours(contour1, contour2, centroidIni, 2)
                    contador+=1
                else:
                    contourDeform = self.deformContours(contour1, contour2, centroidIni, 0)

                area = contourDeform.contourArea()

            #for i in range(len(listVarY)):
            #    randR = random.randint(60, 120)
            #    alphaR = alphaR + randR
            #    randL = random.randint(150, 280)
            #    alphaL = alphaL + randL
            #    if i+1 < len(listVarY):
            #        y1 = listVarY[i]
            #        y2 = listVarY[i+1]
            #        if y1 < y2 :
            #            contour1 = self.resizeContours(resultimage, centroidIni, alphaR, 0)
            #            contour2 = self.resizeContours(resultimage, centroidIni, alphaL, 0)
            #            contourDeform = self.deformContours(contour1, contour2, centroidIni, 0)
            #        else:
            #            contour1 = self.resizeContours(resultimage, centroidIni, alphaR, 1)
            #            contourDeform = self.deformContourDecrease(contour1)

                if area > maximunArea:
                    largestContour = contourDeform
                    maximunArea = area

                polygons = self.generatePolygon(contourDeform)
                for p in polygons.geoms:
                    #print(type(p))
                    multiPolygonList.append(p)

            multiPolygons = MultiPolygon(multiPolygonList)
            lastContour = []
            lastContour.append(largestContour.contour)
            #print("POLYGONS :")
            #print(len(polygons))
            centroidCurrent = centroidIni
            distanceMax = 50
            for centroid in self.centroidList:
                distanceCentroid = self.distanceBetweenTwoPoints(centroidCurrent,centroid)
                print("Centroid :"+str(centroid))
                print("DistanceCentroid :" + str(distanceCentroid))
                if distanceCentroid > distanceMax:
                    centroidCurrent = centroid
                    print("CentroidCurrent :" + str(centroidCurrent))

            raysList = self.generateRays(centroidCurrent, lastContour, numberRays)
            intersections = self.intersectBetweenRaysAndPolygon(multiPolygons, raysList)
            self.drawPoints(intersections)

            pointList1 = self.predictContour(raysList, numberContours)
            #pointList2 = self.predictContour(raysList, 6)
            self.drawContour(pointList1)
            #self.drawContour(pointList2)

            cv2.imwrite("output.jpg", self.image)
            output = cv2.imread("output.jpg")
            cv2.imshow("image",self.image)
            cv2.waitKey(0)
        else:
            print("error loading image")

    def predictContour(self, raysList, time_predict):
        pointList = []
        print("----------Dataset -----------")
        for ray in raysList:
            predicted_point = ray.predictPoint(time_predict)
            pointList.append(predicted_point)
        return pointList

    def drawContour(self, pointList):
        sizeList = len(pointList)
        for i, value in enumerate(pointList):
            if i + 1 < sizeList:
                point1 = pointList[i]
                point2 = pointList[i + 1]
                cv2.circle(self.image, (int(point1[0]), int(point1[1])), 2, self.cianColor, 3)
                cv2.circle(self.image, (int(point2[0]), int(point2[1])), 2, self.cianColor, 3)
                cv2.line(self.image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), self.pinkColor, 2)

        pointIni = pointList[0]
        pointFin = pointList[sizeList - 1]
        cv2.line(self.image, (int(pointIni[0]), int(pointIni[1])), (int(pointFin[0]), int(pointFin[1])), self.pinkColor, 2)

    def resizeContours(self, img, centroid, alpha, mode):
        height = img.shape[0]
        width = img.shape[1]
        if mode == 0:
            resized = imutils.resize(img, width=width + alpha, inter=cv2.INTER_AREA)
        elif mode == 1:
            resized = imutils.resize(img, width=width - alpha, inter=cv2.INTER_AREA)
        # resized = cv2.resize(img,(width+alpha,height),interpolation=cv2.INTER_AREA)
        height = resized.shape[0]
        width = resized.shape[1]
        # cv2.imwrite("imagen"+str(i)+".png",resized)
        _, contours, _ = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        new_centroid = self.findCentroid(contours[0])
        diff_x = centroid[0] - new_centroid[0]
        diff_y = centroid[1] - new_centroid[1]
        self.shiftContour(contours[0], diff_x, diff_y)
        contour = Contour(contours[0], self.blueColor, "increase")
        return contour

    def deformContours(self, contour1, contour2, centroid, mode):
        if mode == 0:
            newContour = self.joinContoursCenter(centroid, contour1, contour2)
        if mode == 1:
            newContour = self.joinContoursLeft(centroid, contour1, contour2)
        if mode == 2:
            newContour = self.joinContoursRigth(centroid, contour1, contour2)
        newContour2 = Contour(np.array(newContour), self.blueColor, "increase")
        return newContour2

    def deformContourDecrease(self, contour1):
        newContour = Contour(np.array(contour1), self.whiteColor, "decrease")
        return newContour

    def deformFuncionParabola(self, a, b, c, rangoX):
        listY = []
        listX = list(range(-rangoX, rangoX))
        print(listX)
        for x in listX:
            ax = a*math.pow(x,2)
            bx = b*x
            y = ax + bx + c
            listY.append(y)
        return listY

    def joinContoursCenter(self, centroid, contour1, contour2):
        aux = True
        newContour = []
        contour1Aux = []
        contour2Aux = []
        contour3Aux = []
        print("GROW CENTER")
        for cnt in contour2.contour:
            if cnt[0][1] < centroid[1]:
                if aux == True:
                    contour1Aux.append([[cnt[0][0], cnt[0][1]]])
                    newContour.append([[cnt[0][0], cnt[0][1]]])
                else:
                    contour3Aux.append([[cnt[0][0], cnt[0][1]]])
            else:
                aux = False
        for cnt in contour1.contour:
            if cnt[0][1] > centroid[1]:
                contour2Aux.append([[cnt[0][0], cnt[0][1]]])
                tupla = (cnt[0][0], cnt[0][1])
                #cv2.circle(self.image, tupla, 3, self.grayColor, 3)
        newContour.extend(contour2Aux)
        newContour.extend(contour3Aux)
        #cv2.drawContours(self.image, contour3Aux, -1, self.pinkColor, 3)
        return newContour

    def joinContoursLeft(self, centroid, contour1, contour2):
        newContour = []
        print("GROW LEFT")
        for cnt in contour2.contour:
            if cnt[0][0] < centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        for cnt in contour1.contour:
            if cnt[0][0] > centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
                tupla = (cnt[0][0], cnt[0][1])
                #cv2.circle(self.image, tupla, 3, self.pinkColor, 3)
        return newContour

    def joinContoursRigth(self, centroid, contour1, contour2):
        newContour = []
        print("GROW RIGTH")
        for cnt in contour1.contour:
            if cnt[0][0] <= centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        for cnt in contour2.contour:
            if cnt[0][0] >= centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        return newContour

    def shiftContour(self, contour, x, y):
        for i, value in enumerate(contour):
            contour[i][0][0] += x
            contour[i][0][1] += y

    def generatePolygon(self, deformContour):
        res = None
        listPolygonsShapely = []

        cnt = deformContour.contour
        color = deformContour.color
        cv2.drawContours(self.image, cnt, -1, color, 3)
        # Contorno nuevo:
        centroid = self.findCentroid(cnt)
        cv2.circle(self.image, centroid, 3, self.cianColor, 3)
        print("Centroide Nuevo: " + str(centroid))
        self.centroidList.append(centroid)

        polygon = []
        startingPoint = (cnt[0][0][0], cnt[0][0][1])
        for pointArray in cnt:
            tupla = (pointArray[0][0], pointArray[0][1])
            if not (pointArray[0][0], pointArray[0][1]) in polygon:
                polygon.append(tupla)
        #polygon.append(startingPoint)
        poligonoShapely = Polygon(polygon)
        print("POLYGON VALID:" + str(poligonoShapely.is_valid))
        #print(explain_validity(poligonoShapely))
        if not poligonoShapely.is_valid:
            #poligonoShapely = geom_factory(lgeos.GEOSMakeValid(poligonoShapely.__geom__))
            poligonoShapely = make_valid(poligonoShapely)
            print("is valid")
            #print(poligonoShapely.is_valid)
            #print(explain_validity(poligonoShapely))
            #print(poligonoShapely)
            if isinstance(poligonoShapely, MultiPolygon):
                ultimo = len(poligonoShapely.geoms)-1
                #print("ultimo " + str(ultimo
                for i in range(ultimo):
                    #print(poligonoShapely.geoms[i])
                    listPolygonsShapely.append(poligonoShapely.geoms[i])
            else:
                if isinstance(poligonoShapely,GeometryCollection):
                    if isinstance(poligonoShapely.geoms[0], MultiPolygon):
                        for geometry in poligonoShapely.geoms[0]:
                            if isinstance(geometry,Polygon):
                                listPolygonsShapely.append(geometry)
                            if isinstance(geometry,MultiPolygon):
                                for polygon in geometry:
                                    listPolygonsShapely.append(polygon)
                    else:
                        listPolygonsShapely.append(poligonoShapely.geoms[0])
                else:
                    listPolygonsShapely.append(poligonoShapely)
            res = MultiPolygon(listPolygonsShapely)
        else:
            listPolygonsShapely.append(poligonoShapely)
            res = MultiPolygon(listPolygonsShapely)
        #print("list polygons shapely")
        #print(listPolygonsShapely)
        return res

    def intersectBetweenRaysAndPolygon(self, polygons, rays):
        listIntersections = []
        for i, polygon in enumerate(polygons.geoms):
            """
            intersection = self.multiLineString.intersection(polygon)
            print("Intersection")
            print(intersection)
            listIntersections.append(intersection)
            """
            for ray in rays:
                intersection = ray.intersect(polygon, i)
                listIntersections.append(intersection)
        return listIntersections

    def segmentRedYellow(self, imageHSV):
        redLowerDown = np.array([0, 100, 100])
        redLowerUp = np.array([10, 255, 255])
        redUpperDown = np.array([160, 100, 100])
        redUpperUp = np.array([179, 255, 255])
        mascara1 = cv2.inRange(imageHSV, redLowerDown, redLowerUp)
        mascara2 = cv2.inRange(imageHSV, redUpperDown, redUpperUp)
        mascaraRed = cv2.addWeighted(mascara1, 1.0, mascara2, 1.0, 0.0)
        imagenRed = cv2.bitwise_and(self.image, self.image, mascaraRed);
        return mascaraRed

    def findCentroid(self, contorno):
        momentos = cv2.moments(contorno)
        huMomentos = cv2.HuMoments(momentos)
        x = 0
        y = 0
        if momentos['m00'] != 0:
            x = int(momentos['m10'] / momentos['m00'])
            y = int(momentos['m01'] / momentos['m00'])
        return (x, y)

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
        if numL <= 8:
            pointsDirection = coords[:numL]
        else:
            pointsDirection = self.calculateMiddlepoints(coords, numL - len(coords))
        self.drawRays(self.image, centroid, pointsDirection)
        rays = self.createRays(centroid, pointsDirection)
        return rays

    def createRays(self, centroid, pointsDirection):
        rays = []
        pointCentroid = Point(centroid)
        for i, point in enumerate(pointsDirection):
            ray = Ray(i, pointCentroid, point)
            lineString = LineString([pointCentroid,point])
            rays.append(ray)
            self.writeImage(str(i), int(point[0]), int(point[1]), self.blackColor)
        return rays

    def calculatePoints(self, listaPoints, num):
        counter = num
        i = 0
        while counter > 0:
            if i + 1 < len(listaPoints):
                p1 = listaPoints[i]
                p2 = listaPoints[i + 1]
                pM = self.middlePoint(p1, p2)
                listaPoints.insert(i + 1, pM)
                i += 2
            else:
                i = 0
            counter -= 1
        return listaPoints

    def writeImage(self, palabra, x, y, color):
        fontFace = cv2.FONT_ITALIC
        fontScale = 0.3
        espesor = 0
        cv2.putText(self.image, palabra, (x, y), fontFace, fontScale, color, espesor)

    def north(self, centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extTop = list(c[c[:, :, 1].argmin()][0])
        extTop[0] = centroid[0]
        res = tuple(extTop)
        return res

    def sub(self, centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extBot = list(c[c[:, :, 1].argmax()][0])
        extBot[0] = centroid[0]
        res = tuple(extBot)
        return res

    def east(self, centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extRight = list(c[c[:, :, 0].argmax()][0])
        extRight[1] = centroid[1]
        res = tuple(extRight)
        return res

    def west(self, centroid, contours):
        c = max(contours, key=cv2.contourArea)
        extLeft = list(c[c[:, :, 0].argmin()][0])
        extLeft[1] = centroid[1]
        res = tuple(extLeft)
        return res

    def puntoMedio(self, coord1, coord2):
        yCoord1 = coord1[1]
        yCoord2 = coord2[1]
        diff = (yCoord2 - yCoord1) / 2
        res = (coord2[0], int(yCoord1 + diff))
        return res

    def drawRays(self, img, centroid, rays):
        for point in rays:
            cv2.circle(img, point, 1, self.whiteColor, 1)
            cv2.line(img, (centroid[0], centroid[1]), (point[0], point[1]), self.whiteColor, 1)

    def drawPoints(self, intersections):
        for list in intersections:
            for intersect in list:
                if isinstance(intersect.intersectionPoint, LineString):
                    if not intersect.intersectionPoint.is_empty:
                        coords = intersect.intersectionPoint.coords;
                        pointIni = coords[0]
                        pointFin = coords[1]
                        point1 = (int(pointIni[0]), int(pointIni[1]))
                        point2 = (int(pointFin[0]), int(pointFin[1]))
                        cv2.circle(self.image, point1, 2, self.yellowColor, 2)
                        cv2.circle(self.image, point2, 2, self.yellowColor, 2)
                else:
                    for intersectionShape in intersect.intersectionPoint.geoms:
                        if isinstance(intersectionShape, LineString):
                            if not intersectionShape.is_empty:
                                coords = intersectionShape.coords;
                                pointIni = coords[0]
                                pointFin = coords[1]
                                point1 = (int(pointIni[0]), int(pointIni[1]))
                                point2 = (int(pointFin[0]), int(pointFin[1]))
                                cv2.circle(self.image, point1, 2, self.yellowColor, 2)
                                cv2.circle(self.image, point2, 2, self.yellowColor, 2)

    def calculateMiddlepoints(self, listaCoords, num):
        counter = num
        i = 0
        while counter > 0:
            if i + 1 < len(listaCoords):
                p1 = listaCoords[i]
                p2 = listaCoords[i + 1]
                pM = self.middlePoint(p1, p2)
                listaCoords.insert(i + 1, pM)
                i += 2
            else:
                p1 = listaCoords[i]
                p2 = listaCoords[0]
                pM = self.middlePoint(p1, p2)
                listaCoords.append(pM)
                i = 0
            counter -= 1
        return listaCoords

    def middlePoint(self, coordA, coordB):
        xA = coordA[0]
        yA = coordA[1]
        xB = coordB[0]
        yB = coordB[1]
        xC = (xA + xB) / 2
        yC = (yA + yB) / 2
        res = (int(xC), int(yC))
        return res

    def distanceBetweenTwoPoints(self, pointA, pointB):
        xA = pointA[0]
        yA = pointA[1]
        xB = pointB[0]
        yB = pointB[1]
        distancia = math.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
        return distancia

    def scaleContour(self, contour, scale, decrease=None):
        M = cv2.moments(contour)
        contourScaled = contour
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            contourNorm = contour - [cx, cy]
            if decrease == None:
                contourScaled = contourNorm * scale
            else:
                contourScaled = contourNorm / scale
            contourScaled = contourScaled + [cx, cy]
            contourScaled = contourScaled.astype(np.int32)
        return contourScaled