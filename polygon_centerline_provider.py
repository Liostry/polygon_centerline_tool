# -*- coding: utf-8 -*-

from qgis.core import QgsProcessingProvider
from PyQt5.QtGui import QIcon
import os
from .polygon_centerline_algorithm import PolygonCenterlineAlgorithm

class PolygonCenterlineProvider(QgsProcessingProvider):
    """Processing Provider для алгоритмов centerline"""

    def __init__(self):
        super().__init__()

    def load(self):
        """Загрузка провайдера"""
        self.refreshAlgorithms()
        return True

    def unload(self):
        """Выгрузка провайдера"""
        pass

    def loadAlgorithms(self):
        """Загрузка алгоритмов"""
        self.addAlgorithm(PolygonCenterlineAlgorithm())

    def id(self):
        """Уникальный ID провайдера"""
        return 'polygoncenterline'

    def name(self):
        """Человекочитаемое имя"""
        return 'Polygon Centerline'

    def longName(self):
        """Длинное имя для отображения"""
        return self.name()

    def icon(self):
        """Иконка провайдера"""
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)
