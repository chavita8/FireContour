
class Segmento(object):
    def __init__(self, contornoID, segmentoID, centroide, punto, distancia):
        self.contornoID = contornoID
        self.segmentoID = segmentoID
        self.centroide = centroide
        self.punto = punto
        self.distancia = distancia

    def contornoID(self):
        return self.contornoID

    def segmentoID(self):
        return self.segmentoID

    def centroide(self):
        return self.centroide

    def punto(self):
        return self.punto

    def distancia(self):
        return self.distancia