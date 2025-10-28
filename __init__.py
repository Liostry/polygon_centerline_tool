# -*- coding: utf-8 -*-

def classFactory(iface):
    from .polygon_centerline_plugin import PolygonCenterlinePlugin
    return PolygonCenterlinePlugin(iface)
