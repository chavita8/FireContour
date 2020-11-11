
class Segmento(object):
    def __init__(self, imagenID, segmentoID, centroide, punto, distancia):
        self.imagenID = imagenID
        self.segmentoID = segmentoID
        self.centroide = centroide
        self.punto = punto
        self.distancia = distancia

    def imagenID(self):
        return self.imagenID

    def segmentoID(self):
        return self.segmentoID

    def centroide(self):
        return self.centroide

    def punto(self):
        return self.punto

    def distanci(self):
        return self.distancia