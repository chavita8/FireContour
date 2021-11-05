import cv2
import numpy as np
import math
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, MultiLineString, LineString, LinearRing, GeometryCollection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from contour import Contour
import random
import time
import csv
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
        self.contoursList = []
        self.contoursListCenter = []
        self.contoursListLeft = []
        self.contoursListRight = []
        self.contoursListDeform = []
        self.multiLineString = []

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
            centroid = self.findCentroid(contours[0])
            cv2.circle(self.image, centroid, 3, self.whiteColor, 3)

            self.resizeContoursEvents(resultimage, numberContours, centroid)
            # self.resizeContoursCenter(resultimage,numberContours,centroid)
            # self.resizeContoursLeft(resultimage,numberContours,centroid)
            # self.resizeContoursRigth(resultimage,numberContours,centroid)

            polygons, res = self.generatePolygons()
            lastContour = []
            lastContour.append(res)
            raysList = self.generateRays(centroid, lastContour, numberRays)
            intersections = self.intersectBetweenRaysAndPolygon(polygons, raysList)

            #rayId = 1
            #ray = raysList[rayId]
            #distances = ray.getDistances()
            #print('\n\n DISTANCES RAY ONE: ')
            #print(len(distances))
            #print(distances)
            #rayId = 5
            #ray = raysList[rayId]
            #distances = ray.getDistances()
            #print('\n\n DISTANCES RAY TWO: ')
            #print(len(distances))
            #print(distances)
            for rayId in range(numberRays):
                ray = raysList[rayId]
                listIntersections = ray.getIntersections()
                print("Intersections :"+ str(listIntersections))
                destinationPoint = ray.getDestinationPoint()
                print("DestinationPoint :"+ str(destinationPoint))
                originPoint = ray.getOriginPoint()
                print("OriginPoint :"+ str(originPoint))
                distances = ray.getDistances()
                self.generarCSV(distances, rayId)
            self.drawPoints(intersections)
            #plt.plot(distances)
            #plt.show()
            print("MultiLineString")
            print(self.multiLineString)
            cv2.imwrite("output.jpg", self.image)
            output = cv2.imread("output.jpg")
            cv2.imshow("image",self.image)
            cv2.waitKey(0)
        else:
            print("error loading image")

    def resizeContoursEvents(self, img, numberContours, centroid):
        contours = numberContours * 2
        alphaL = 30
        alphaR = 0
        for i in range(contours):
            if i % 2 != 0:
                randL = random.randint(150, 280)
                alphaL = alphaL + randL
                self.resizeContours(img, centroid, alphaL)
            else:
                randR = random.randint(60, 120)
                alphaR = alphaR + randR
                self.resizeContours(img, centroid, alphaR)
        self.deformContours(centroid)

    def resizeContours(self, img, centroide, alpha):
        contours_list = []
        height = img.shape[0]
        width = img.shape[1]
        print("centroide inicial: " + str(centroide))
        resized = imutils.resize(img, width=width + alpha, inter=cv2.INTER_AREA)
        # resized = cv2.resize(img,(width+alpha,height),interpolation=cv2.INTER_AREA)
        height = resized.shape[0]
        width = resized.shape[1]
        # cv2.imwrite("imagen"+str(i)+".png",resized)
        _, contours, _ = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_list.append(contours)
        new_centroid = self.findCentroid(contours[0])
        diff_x = centroide[0] - new_centroid[0]
        diff_y = centroide[1] - new_centroid[1]
        self.shiftContour(contours[0], diff_x, diff_y)
        print("nuevo centroide: " + str(new_centroid))
        # print("diferencia centroide: " + str((diff_x, diff_y)))
        contour = Contour(contours[0], self.blueColor, "increase")
        self.contoursList.append(contour)

    def deformContours(self, centroid):
        contoursListCurrent = self.contoursList
        cant = len(contoursListCurrent)
        limite = cant/3
        contador = 0
        aux = 0
        for cnt in contoursListCurrent:
            if aux + 1 < len(contoursListCurrent):
                contour1 = contoursListCurrent[aux]
                contour2 = contoursListCurrent[aux + 1]
                if contador >= 0 and contador <= limite:
                    newContour = self.joinContoursCenter(centroid, contour1, contour2)
                if contador > limite and contador <= limite*2:
                    newContour = self.joinContoursLeft(centroid, contour1, contour2)
                if contador > limite*2 and contador <= limite*3:
                    newContour = self.joinContoursRigth2(centroid, contour1, contour2)

                newContour2 = Contour(np.array(newContour), self.blueColor, "increase")
                self.contoursListDeform.append(newContour2)
            aux = aux + 2
            contador += 1

    def joinContoursCenter(self, centroid, contour1, contour2):
        aux = True
        newContour = []
        contour1Aux = []
        contour2Aux = []
        contour3Aux = []
        #print("CENTROID :"+str(centroid))
        #print(centroid[0])
        #print(centroid[1]+100)
        #print(centroid[1]-100)
        #tupla1 = (centroid[0],centroid[1]+150)
        #tupla2 = (centroid[0],centroid[1]-300)
        #cv2.circle(self.image, tupla1, 3, self.pinkColor, 3)
        #cv2.circle(self.image, tupla2, 3, self.cianColor, 3)
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
        newContour.extend(contour2Aux)
        newContour.extend(contour3Aux)
        return newContour

    def joinContoursLeft(self, centroid, contour1, contour2):
        newContour = []
        print("centroide " + str(centroid))
        for cnt in contour2.contour:
            if cnt[0][0] < centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        for cnt in contour1.contour:
            if cnt[0][0] > centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        return newContour

    def joinContoursRigth2(self, centroid, contour1, contour2):
        newContour = []
        print("centroide " + str(centroid))
        for cnt in contour1.contour:
            if cnt[0][0] <= centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        for cnt in contour2.contour:
            if cnt[0][0] >= centroid[0]:
                newContour.append([[cnt[0][0], cnt[0][1]]])
        return newContour

    def joinContoursRigth(self, centroid, contour1, contour2):
        aux = 0
        var = 0
        extreme = []
        newContour = []
        contour1Aux = []
        contour2Aux = []
        centroid = (centroid[0] - 8, centroid[1])
        print("contour2: " + str(contour2.contour))
        for cnt in contour1.contour:
            if cnt[0][0] <= centroid[0]:
                contour1Aux.append([[cnt[0][0], cnt[0][1]]])
                #newContour.append([[cnt[0][0], cnt[0][1]]])
                # print("aux "+ str(aux))
                # if aux == 15:
                #  cv2.circle(self.image, (cnt[0][0],cnt[0][1]), 3, self.whiteColor, 3)
                # if aux == 450:
                #  cv2.circle(self.image, (cnt[0][0],cnt[0][1]), 3, self.whiteColor, 3)
            aux = aux + 1
        for cnt in contour2.contour:
            print("point: " + str([[cnt[0][0], cnt[0][1]]]))
            if cnt[0][0] >= centroid[0]:
                #newContour.append([[cnt[0][0], cnt[0][1]]])
                contour2Aux.append([[cnt[0][0], cnt[0][1]]])
                if cnt[0][0] == centroid[0] and cnt[0][1] < centroid[1]:
                    inicio = [cnt[0][0], cnt[0][1]]
                if cnt[0][0] == centroid[0] and cnt[0][1] > centroid[1]:
                    extreme = [cnt[0][0], cnt[0][1]]
            var = var + 1
        # Contour2
        lenContour2Aux = len(contour2Aux)
        ultimaPos = lenContour2Aux - 1
        p1Contour2 = inicio
        p2Contour2 = extreme
        # print("extremo1"+str(type(p1Contour2)))
        # print("extremo2"+str(type(p2Contour2)))
        # Contour1
        lenContour1Aux = len(contour1Aux)
        ultimaPos1 = lenContour1Aux - 1
        p1Contour1 = contour1Aux[0][0]
        p2Contour1 = contour1Aux[ultimaPos1][0]
        # print("extremo1 C1 "+str(p1Contour1))
        # print("extremo2 C1 "+str(p2Contour1))
        listaPoints1 = [p1Contour2, p1Contour1]
        listaPoints2 = [p2Contour1, p2Contour2]
        distance = int(self.distanceBetweenTwoPoints(p1Contour2, p1Contour1))
        distance2 = int(self.distanceBetweenTwoPoints(p2Contour1, p2Contour2))
        listaNew = self.calculatePoints(listaPoints1, distance * 2)
        listaNew2 = self.calculatePoints(listaPoints2, distance2 * 2)
        for cnt in contour1Aux:
            newContour.append(cnt)
        for point in listaNew2:
            if not ([[point[0],point[1]]]) in newContour:
                newContour.append(([[point[0],point[1]]]))
        for cnt in contour2Aux:
            if not cnt in newContour:
                newContour.append(cnt)
        for point in listaNew:
            if not ([[point[0],point[1]]]) in newContour:
                newContour.append(([[point[0],point[1]]]))
        # print(("NEW CONTOUR: ") + str(len(newContour)))
        # print(newContour)
        return newContour

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

    def shiftContour(self, contour, x, y):
        for i, value in enumerate(contour):
            contour[i][0][0] += x
            contour[i][0][1] += y

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

    def generatePolygons(self):
        contoursListP = self.contoursListDeform
        listPolygonsShapely = []
        largestContour = self.largestContour(contoursListP)

        for contourObj in contoursListP:
            cnt = contourObj.contour
            color = contourObj.color
            cv2.drawContours(self.image, cnt, -1, color, 3)
            polygon = []
            array = cnt
            startingPoint = (array[0][0][0], array[0][0][1])
            for pointArray in array:
                tupla = (pointArray[0][0], pointArray[0][1])
                if not (pointArray[0][0], pointArray[0][1]) in polygon:
                    polygon.append(tupla)
            #polygon.append(startingPoint)
            poligonoShapely = Polygon(polygon)
            print("POLYGON VALID:" + str(poligonoShapely.is_valid))
            print(explain_validity(poligonoShapely))
            if not poligonoShapely.is_valid:
                #poligonoShapely = geom_factory(lgeos.GEOSMakeValid(poligonoShapely.__geom__))
                poligonoShapely = make_valid(poligonoShapely)
                print("is valid")
                print(poligonoShapely.is_valid)
                print(explain_validity(poligonoShapely))
                print(poligonoShapely)
                if isinstance(poligonoShapely, MultiPolygon):
                    ultimo = len(poligonoShapely.geoms)-1
                    print("ultimo " + str(ultimo))
                    for i in range(ultimo):
                        print(poligonoShapely.geoms[i])
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
            else:
                listPolygonsShapely.append(poligonoShapely)
        print("list polygons shapely")
        print(listPolygonsShapely)
        multiPolygon = MultiPolygon(listPolygonsShapely)
        return (multiPolygon, largestContour.contour)

    def largestContour(self, contoursList):
        largest = None
        maximunArea = -1
        for contour in contoursList:
            area = contour.contourArea()
            if area > maximunArea:
                largest = contour
                maximunArea = area
        return largest

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
            self.multiLineString.append(lineString)
            rays.append(ray)
            self.writeImage(str(i), int(point[0]), int(point[1]), self.blackColor)
        self.multiLineString = MultiLineString(self.multiLineString)
        return rays

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
            """
            for intersection in list:
                point = intersection.intersectionPoint
                if isinstance(point, Point):
                    x = point.x
                    y = point.y
                    cv2.circle(self.image, (int(x), int(y)), 2, self.yellowColor, 2)
                    self.writeImage(str(intersection.contourId), int(x), int(y), self.cianColor)
            """
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

    def generarCSV(self, distances, rayID):
        print('\n\n DISTANCES CSV: ')
        print(len(distances))
        times = np.array(range(0, len(distances)))
        # times = np.linspace(0.1, 50.0, len(distances))
        X = times.reshape(-1, 1)
        Y = np.array(distances).reshape(-1, 1)
        csv_arr = []
        csv_arr.append(["tiempo", "distancia"])
        for i, value in enumerate(X):
            arr = []
            x = X[i]
            y = Y[i]
            arr.append(x[0])
            arr.append(y[0])
            csv_arr.append(arr)
        filename = 'distanciasRayo' + str(rayID) + '.csv'
        myFile = open(filename, 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(csv_arr)
        # df = pd.read_csv(filename)s
