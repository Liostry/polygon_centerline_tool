# -*- coding: utf-8 -*-

from qgis.core import QgsApplication
from .polygon_centerline_provider import PolygonCenterlineProvider

class PolygonCenterlinePlugin:
    """Главный класс плагина"""
    
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Инициализация Processing Provider"""
        self.provider = PolygonCenterlineProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Инициализация GUI (обязательный метод)"""
        self.initProcessing()

    def unload(self):
        """Выгрузка плагина (обязательный метод)"""
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
