# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterField,
                       QgsProcessingParameterString,
                       QgsProcessingParameterEnum,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsFields,
                       QgsField,
                       QgsFeatureRequest)
from qgis import processing
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import (Polygon, MultiPolygon, LineString, 
                              MultiLineString, Point)
from shapely.ops import linemerge, unary_union, nearest_points
from shapely import wkt
from shapely.validation import make_valid, explain_validity

# -- Блок дефолтных параметров (ЕДИНОЕ МЕСТО ДЛЯ НАСТРОЕК) --
DEFAULTS = {
    'DENSITY': 1.0,           # meters
    'SIMPLIFY': 0.1,         # meters
    'SIMPLIFY_MIN': 0.0,
    'SIMPLIFY_MAX': 10.0,
    'DENSITY_MIN': 0.01,
    'DENSITY_MAX': 100.0,
    'SKELETON_DEFAULT': 1,   # Главная ось
    'SPLIT_AT_JUNCTIONS': False,
    'CONNECT_FEATURES': False,
    'FIX_GEOMETRIES': True
}

# -- ЗАЩИТА ОТ ОТСУТСТВИЯ ЗАВИСИМОСТЕЙ --
missing_dependencies = []
try:
    import numpy as np
except ImportError:
    np = None
    missing_dependencies.append('numpy')
try:
    from scipy.spatial import Voronoi
except ImportError:
    Voronoi = None
    missing_dependencies.append('scipy (scipy.spatial.Voronoi)')
try:
    from shapely.geometry import (Polygon, MultiPolygon, LineString, MultiLineString, Point)
    from shapely.ops import linemerge, unary_union, nearest_points
    from shapely import wkt
    from shapely.validation import make_valid, explain_validity
except ImportError:
    Polygon = None
    MultiPolygon = None
    LineString = None
    MultiLineString = None
    Point = None
    linemerge = unary_union = nearest_points = wkt = make_valid = explain_validity = None
    missing_dependencies.append('shapely')

class PolygonCenterlineAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    DENSITY = 'DENSITY'
    SIMPLIFY = 'SIMPLIFY_TOLERANCE'
    CONNECT_FEATURES = 'CONNECT_FEATURES'
    SPLIT_AT_JUNCTIONS = 'SPLIT_AT_JUNCTIONS'
    FILTER_FIELD = 'FILTER_FIELD'
    FILTER_VALUES = 'FILTER_VALUES'
    FIX_GEOMETRIES = 'FIX_GEOMETRIES'
    SKELETON_MODE = 'SKELETON_MODE'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return PolygonCenterlineAlgorithm()

    def name(self):
        return 'polygontocenterline'

    def displayName(self):
        return self.tr('Polygon to Centerline')

    def group(self):
        return self.tr('Vector geometry')

    def groupId(self):
        return 'vectorgeometry'

    def shortHelpString(self):
        return self.tr("""
        Создает центральные линии из полигональных объектов методом диаграммы Вороного.
        
        Режимы построения:
        - Полный скелет: все ветви медиальной оси (для сложных форм)
        - Главная ось: упрощение до одной линии от торца до торца (рекомендуется для парковок/дорог)
        - Только самая длинная: берётся самый длинный сегмент
        
        Использование фильтра:
        1. Выберите поле для фильтрации
        2. Оставьте значения пустыми → запустите → увидите все доступные значения
        3. Введите значения через запятую
        4. Запустите снова с фильтром
        """)

    def initAlgorithm(self, config=None):
        """
        Инициализация параметров QGIS Processing Algorithm.
        Все диапазоны и дефолтные значения берутся из DEFAULTS.
        """
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Входные полигоны'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.FILTER_FIELD,
                self.tr('Поле для фильтрации (необязательно)'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
                optional=True,
                allowMultiple=False
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.FILTER_VALUES,
                self.tr('Значения для фильтрации (через запятую, пусто = показать все)'),
                optional=True,
                defaultValue='',
                multiLine=False
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FIX_GEOMETRIES,
                self.tr('Автоматически исправлять некорректные геометрии'),
                defaultValue=DEFAULTS['FIX_GEOMETRIES']
            )
        )

        # НОВЫЙ ПАРАМЕТР: Режим построения
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SKELETON_MODE,
                self.tr('Режим построения'),
                options=[
                    self.tr('Полный скелет (все ветви)'),
                    self.tr('Главная ось (от торца до торца)'),
                    self.tr('Только самая длинная линия')
                ],
                defaultValue=DEFAULTS['SKELETON_DEFAULT'],  # Главная ось
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.DENSITY,
                self.tr('Плотность дискретизации границы (метры)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULTS['DENSITY'],
                minValue=DEFAULTS['DENSITY_MIN'],
                maxValue=DEFAULTS['DENSITY_MAX']
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SIMPLIFY,
                self.tr('Допуск упрощения (0 = без упрощения)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULTS['SIMPLIFY'],
                minValue=DEFAULTS['SIMPLIFY_MIN'],
                maxValue=DEFAULTS['SIMPLIFY_MAX']
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CONNECT_FEATURES,
                self.tr('Соединять касающиеся полигоны'),
                defaultValue=DEFAULTS['CONNECT_FEATURES']
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SPLIT_AT_JUNCTIONS,
                self.tr('Разделять на сегменты в точках пересечения'),
                defaultValue=DEFAULTS['SPLIT_AT_JUNCTIONS']  # Для главной оси лучше False
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Центральные линии')
            )
        )

    def densify_boundary(self, polygon, density):
        """
        Дискретизация границы полигона равномерно, чтобы подготовить вход к Voronoi.
        :param polygon: Объект shapely Polygon
        :param density: Шаг в метрах между точками
        :return: np.array точек по периметру
        """
        coords = []
        boundary = polygon.boundary
        
        if boundary.geom_type == 'LineString':
            lines = [boundary]
        else:
            lines = list(boundary.geoms)
        
        for line in lines:
            line_coords = list(line.coords)
            for i in range(len(line_coords) - 1):
                p1 = np.array(line_coords[i])
                p2 = np.array(line_coords[i + 1])
                dist = np.linalg.norm(p2 - p1)
                
                if dist > density:
                    num_points = int(np.ceil(dist / density))
                    for j in range(num_points):
                        t = j / num_points
                        point = p1 + t * (p2 - p1)
                        coords.append(tuple(point))
                else:
                    coords.append(tuple(p1))
            
            coords.append(tuple(line_coords[-1]))
        
        return np.array(list(dict.fromkeys(coords)))

    def create_centerline_voronoi(self, polygon, density):
        """
        Строит центрлайн как медиальную ось с помощью Voronoi диаграммы для точек границы полигона.
        :param polygon: shapely Polygon
        :param density: float, шаг дискретизации
        :return: LineString или MultiLineString (shapely)
        """
        try:
            points = self.densify_boundary(polygon, density)
            
            if len(points) < 4:
                return None
            
            vor = Voronoi(points)
            
            centerline_segments = []
            for ridge_vertices in vor.ridge_vertices:
                if -1 not in ridge_vertices:
                    p1 = vor.vertices[ridge_vertices[0]]
                    p2 = vor.vertices[ridge_vertices[1]]
                    
                    mid_point = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                    if polygon.contains(Point(p1)) and polygon.contains(Point(p2)):
                        centerline_segments.append(LineString([p1, p2]))
            
            if not centerline_segments:
                return None
            
            merged = linemerge(centerline_segments)
            return merged
            
        except Exception as e:
            raise QgsProcessingException(f'Ошибка Voronoi: {str(e)}')

    def simplify_skeleton_to_main_path(self, centerline, polygon, feedback=None):
        """
        Пытается выбрать главную ветвь медиальной оси и объединить фрагменты,
        чтобы получить максимально длинную и непрерывную ось.
        :param centerline: LineString/MultiLineString (shapely)
        :param polygon: исходный полигон для mask
        :return: LineString
        """
        try:
            # Если это одна линия - возвращаем как есть
            if centerline.geom_type == 'LineString':
                return centerline
            
            # Если MultiLineString - обрабатываем
            if centerline.geom_type == 'MultiLineString':
                lines = list(centerline.geoms)
                
                if len(lines) == 1:
                    return lines[0]
                
                # Сортируем по длине (самая длинная первая)
                lines_sorted = sorted(lines, key=lambda x: x.length, reverse=True)
                main_line = lines_sorted[0]
                
                if feedback:
                    feedback.pushInfo(f'    Найдено {len(lines)} сегментов, главная линия: {main_line.length:.2f}м')
                
                # Если главная линия значительно длиннее остальных - берём только её
                if len(lines_sorted) > 1:
                    length_ratio = main_line.length / lines_sorted[1].length
                    if length_ratio > 2.0:
                        if feedback:
                            feedback.pushInfo(f'    Главная линия в {length_ratio:.1f}x длиннее, используем только её')
                        return main_line
                
                # Пытаемся соединить близкие линии последовательно
                connected = self.connect_lines_sequentially(lines_sorted, polygon)
                return connected
            
            return centerline
            
        except Exception as e:
            if feedback:
                feedback.pushWarning(f'    Ошибка упрощения скелета: {str(e)}')
            return centerline

    def connect_lines_sequentially(self, lines, polygon):
        """Соединение линий в последовательную цепочку"""
        if len(lines) == 1:
            return lines[0]
        
        # Начинаем с самой длинной линии
        main_coords = list(lines[0].coords)
        remaining_lines = lines[1:]
        
        # Максимальное расстояние для соединения (5% от периметра)
        max_gap = polygon.length * 0.05
        
        connected_any = True
        while connected_any and remaining_lines:
            connected_any = False
            
            start_point = Point(main_coords[0])
            end_point = Point(main_coords[-1])
            
            for i, line in enumerate(remaining_lines):
                line_start = Point(line.coords[0])
                line_end = Point(line.coords[-1])
                
                # Проверяем все варианты соединения
                if start_point.distance(line_start) < max_gap:
                    main_coords = list(reversed(line.coords[:-1])) + main_coords
                    remaining_lines.pop(i)
                    connected_any = True
                    break
                elif start_point.distance(line_end) < max_gap:
                    main_coords = list(line.coords[:-1]) + main_coords
                    remaining_lines.pop(i)
                    connected_any = True
                    break
                elif end_point.distance(line_start) < max_gap:
                    main_coords = main_coords + list(line.coords[1:])
                    remaining_lines.pop(i)
                    connected_any = True
                    break
                elif end_point.distance(line_end) < max_gap:
                    main_coords = main_coords + list(reversed(line.coords[1:]))
                    remaining_lines.pop(i)
                    connected_any = True
                    break
        
        return LineString(main_coords)

    def extend_to_boundaries(self, centerline, polygon, feedback=None):
        """
        Экстраполирует (продлевает) начало и конец линии до ближайшей кромки (границы) полигона.
        :return: LineString c новыми координатами концов
        """
        try:
            if centerline.geom_type != 'LineString':
                return centerline
            
            coords = list(centerline.coords)
            
            if len(coords) < 2:
                return centerline
            
            boundary = polygon.boundary
            
            # === НАЧАЛО ЛИНИИ ===
            start = np.array(coords[0])
            second = np.array(coords[min(1, len(coords)-1)])
            
            # Вектор направления в начале (от второй точки к первой)
            start_direction = start - second
            start_direction_norm = np.linalg.norm(start_direction)
            
            if start_direction_norm > 0:
                start_direction = start_direction / start_direction_norm
                
                # Продляем начало до границы
                max_extension = polygon.length * 0.5  # Максимум 50% периметра
                step = max_extension / 200  # 200 шагов
                
                best_point = start
                for i in range(1, 201):
                    test_point = start + start_direction * (step * i)
                    test_geom = Point(test_point)
                    
                    if not polygon.contains(test_geom):
                        # Точка вышла за границу, берём предыдущую
                        break
                    best_point = test_point
                
                coords[0] = tuple(best_point)
            
            # === КОНЕЦ ЛИНИИ ===
            end = np.array(coords[-1])
            pre_end = np.array(coords[max(-2, -len(coords))])
            
            # Вектор направления в конце (от предпоследней к последней)
            end_direction = end - pre_end
            end_direction_norm = np.linalg.norm(end_direction)
            
            if end_direction_norm > 0:
                end_direction = end_direction / end_direction_norm
                
                max_extension = polygon.length * 0.5
                step = max_extension / 200
                
                best_point = end
                for i in range(1, 201):
                    test_point = end + end_direction * (step * i)
                    test_geom = Point(test_point)
                    
                    if not polygon.contains(test_geom):
                        break
                    best_point = test_point
                
                coords[-1] = tuple(best_point)
            
            if feedback:
                original_length = centerline.length
                new_line = LineString(coords)
                extension = new_line.length - original_length
                if extension > 0.1:
                    feedback.pushInfo(f'    Продлено на {extension:.2f}м (было {original_length:.2f}м → стало {new_line.length:.2f}м)')
            
            return LineString(coords)
            
        except Exception as e:
            if feedback:
                feedback.pushWarning(f'    Ошибка продления: {str(e)}')
            return centerline

    def split_at_junctions(self, geom):
        """Разделение линий в точках пересечения"""
        if geom.geom_type == 'LineString':
            return [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
            union = unary_union(lines)
            if union.geom_type == 'LineString':
                return [union]
            else:
                return list(union.geoms)
        return [geom]

    def fix_geometry(self, shapely_geom, feature_id, feedback):
        """
        Исправление самопересечений и иных ошибок геометрии через make_valid.
        :return: Исправленный shapely Polygon или None
        """
        try:
            if not shapely_geom.is_valid:
                reason = explain_validity(shapely_geom)
                feedback.pushInfo(f'  Объект {feature_id}: {reason}')
                
                fixed = make_valid(shapely_geom)
                
                if fixed.geom_type == 'GeometryCollection':
                    polys = [g for g in fixed.geoms 
                            if g.geom_type in ['Polygon', 'MultiPolygon']]
                    if not polys:
                        feedback.pushWarning(f'  ✗ Нет полигонов после исправления')
                        return None
                    fixed = unary_union(polys)
                
                if fixed.is_valid:
                    feedback.pushInfo(f'  ✓ Исправлено: {shapely_geom.geom_type} → {fixed.geom_type}')
                    return fixed
                else:
                    feedback.pushWarning(f'  ✗ Геометрия всё ещё некорректна')
                    return None
            
            return shapely_geom
            
        except Exception as e:
            feedback.pushWarning(f'  ✗ Ошибка исправления: {str(e)}')
            return None

    def get_unique_values(self, source, field_name, feedback):
        """Получение всех уникальных значений поля"""
        field_idx = source.fields().indexOf(field_name)
        if field_idx < 0:
            return set()
        
        unique_vals = {}
        total = source.featureCount()
        
        feedback.pushInfo('Сканирование уникальных значений...')
        
        for i, feat in enumerate(source.getFeatures()):
            if i % 1000 == 0:
                feedback.setProgress(int(i * 50 / total))
            
            val = feat.attribute(field_name)
            if val is not None:
                val_str = str(val)
                unique_vals[val_str] = unique_vals.get(val_str, 0) + 1
        
        return unique_vals

    def processAlgorithm(self, parameters, context, feedback):
        """
        Основная логика обработки: для каждого (мульти)полигона строит центрлайн методом Voronoi,
        корректирует геометрию и пишет результат в sink. Для больших данных использует итераторы,
        избегает хранения больших списков в памяти. Логирует ошибки отсутствия зависимостей.
        """
        # Проверка зависимостей
        if missing_dependencies:
            deps = ', '.join(missing_dependencies)
            feedback.reportError(
                'ВНИМАНИЕ: Для работы алгоритма не хватает зависимостей: ' + deps +\
                '\n\nУстановите их в окружение QGIS (OSGeo4W или через pip).')
            raise QgsProcessingException('Отсутствуют зависимости: ' + deps)
        if np is None or Voronoi is None or Polygon is None:
            raise QgsProcessingException('Ошибка инициализации зависимостей numpy/scipy/shapely')
        
        source = self.parameterAsSource(parameters, self.INPUT, context)
        density = self.parameterAsDouble(parameters, self.DENSITY, context)
        simplify = self.parameterAsDouble(parameters, self.SIMPLIFY, context)
        connect = self.parameterAsBoolean(parameters, self.CONNECT_FEATURES, context)
        split = self.parameterAsBoolean(parameters, self.SPLIT_AT_JUNCTIONS, context)
        filter_field = self.parameterAsString(parameters, self.FILTER_FIELD, context)
        filter_values_str = self.parameterAsString(parameters, self.FILTER_VALUES, context)
        fix_geom = self.parameterAsBoolean(parameters, self.FIX_GEOMETRIES, context)
        skeleton_mode = self.parameterAsEnum(parameters, self.SKELETON_MODE, context)

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        # Показ доступных значений
        if filter_field and not filter_values_str:
            unique_vals = self.get_unique_values(source, filter_field, feedback)
            
            feedback.pushInfo('')
            feedback.pushInfo('=' * 80)
            feedback.pushInfo(f'📋 ДОСТУПНЫЕ ЗНАЧЕНИЯ В ПОЛЕ "{filter_field}":')
            feedback.pushInfo('=' * 80)
            feedback.pushInfo(f'{"Значение":<50} {"Количество":>15}')
            feedback.pushInfo('-' * 80)
            
            for val in sorted(unique_vals.keys()):
                count = unique_vals[val]
                feedback.pushInfo(f'{val:<50} {count:>15}')
            
            feedback.pushInfo('=' * 80)
            feedback.pushInfo(f'Всего уникальных значений: {len(unique_vals)}')
            feedback.pushInfo(f'Всего объектов: {sum(unique_vals.values())}')
            feedback.pushInfo('=' * 80)
            feedback.pushInfo('')
            feedback.pushInfo('💡 Введите нужные значения через запятую и запустите снова')
            feedback.pushInfo('')
            
            return {}

        # Создание выходных полей
        fields = QgsFields()
        fields.append(QgsField('fid_source', QVariant.Int))
        fields.append(QgsField('length_m', QVariant.Double))
        
        for field in source.fields():
            fields.append(field)

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            QgsWkbTypes.LineString,
            source.sourceCrs()
        )

        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        # Построение фильтра
        request = QgsFeatureRequest()
        if filter_field and filter_values_str:
            values = [v.strip() for v in filter_values_str.split(',')]
            
            expressions = []
            for val in values:
                val_escaped = val.replace("'", "''")
                expressions.append(f'"{filter_field}" = \'{val_escaped}\'')
            
            expression = ' OR '.join(expressions)
            request.setFilterExpression(expression)
            
            feedback.pushInfo('')
            feedback.pushInfo('🔍 Применён фильтр:')
            feedback.pushInfo(f'   {expression}')
            feedback.pushInfo('')
        
        # Название режима
        mode_names = ['Полный скелет', 'Главная ось', 'Самая длинная линия']
        feedback.pushInfo(f'📐 Режим построения: {mode_names[skeleton_mode]}')
        feedback.pushInfo('')
        
        total = source.featureCount()
        filtered_count = 0
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Получение объектов
        if connect:
            feedback.pushInfo('Объединение касающихся полигонов...')
            
            temp_layer = processing.run("native:extractbyexpression", {
                'INPUT': parameters[self.INPUT],
                'EXPRESSION': request.filterExpression().expression() if request.filterExpression() else 'True',
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)
            
            dissolved = processing.run("native:dissolve", {
                'INPUT': temp_layer['OUTPUT'],
                'FIELD': [],
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)
            
            features = dissolved['OUTPUT'].getFeatures()
            total = dissolved['OUTPUT'].featureCount()
        else:
            features = source.getFeatures(request)
            if filter_field and filter_values_str:
                total = sum(1 for _ in source.getFeatures(request))

        feedback.pushInfo('🚀 Начало обработки...')
        feedback.pushInfo('')

        # Обработка
        for current, feature in enumerate(features):
            if feedback.isCanceled():
                break

            filtered_count += 1
            geom = feature.geometry()
            
            if geom.isEmpty() or geom.isNull():
                feedback.pushWarning(f'Объект {feature.id()}: пустая геометрия')
                skipped_count += 1
                continue

            try:
                shapely_geom = wkt.loads(geom.asWkt())
                
                if fix_geom:
                    shapely_geom = self.fix_geometry(shapely_geom, feature.id(), feedback)
                    if shapely_geom is None:
                        error_count += 1
                        continue
                
                if shapely_geom.geom_type == 'Polygon':
                    polygons = [shapely_geom]
                elif shapely_geom.geom_type == 'MultiPolygon':
                    polygons = list(shapely_geom.geoms)
                else:
                    feedback.pushWarning(f'Объект {feature.id()}: неподдерживаемый тип {shapely_geom.geom_type}')
                    skipped_count += 1
                    continue

                all_lines = []
                
                for poly in polygons:
                    if fix_geom and not poly.is_valid:
                        poly = make_valid(poly)
                    
                    if not poly.is_valid:
                        feedback.pushWarning(f'Объект {feature.id()}: некорректная геометрия')
                        continue
                    
                    # Создание centerline
                    centerline = self.create_centerline_voronoi(poly, density)
                    
                    if centerline is None:
                        continue
                    
                    # === ПРИМЕНЕНИЕ РЕЖИМА ===
                    if skeleton_mode == 1:  # Главная ось
                        centerline = self.simplify_skeleton_to_main_path(centerline, poly, feedback)
                        centerline = self.extend_to_boundaries(centerline, poly, feedback)
                    elif skeleton_mode == 2:  # Только самая длинная
                        if centerline.geom_type == 'MultiLineString':
                            lines = list(centerline.geoms)
                            centerline = max(lines, key=lambda x: x.length)
                            feedback.pushInfo(f'    Выбрана самая длинная из {len(lines)} линий: {centerline.length:.2f}м')
                        centerline = self.extend_to_boundaries(centerline, poly, feedback)
                    # skeleton_mode == 0: полный скелет, не изменяем
                    
                    # Упрощение геометрии
                    if simplify > 0:
                        centerline = centerline.simplify(simplify, preserve_topology=True)
                    
                    # Разделение на сегменты
                    if split:
                        lines = self.split_at_junctions(centerline)
                    else:
                        if centerline.geom_type == 'LineString':
                            lines = [centerline]
                        else:
                            lines = list(centerline.geoms)
                    
                    all_lines.extend(lines)

                # Создание выходных объектов
                for line in all_lines:
                    if line.length > 0:
                        qgs_geom = QgsGeometry.fromWkt(line.wkt)
                        
                        out_feature = QgsFeature()
                        out_feature.setGeometry(qgs_geom)
                        
                        attrs = [feature.id(), line.length] + feature.attributes()
                        out_feature.setAttributes(attrs)
                        
                        sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
                        processed_count += 1

            except Exception as e:
                feedback.pushWarning(f'Объект {feature.id()}: {str(e)}')
                error_count += 1
                import traceback
                feedback.pushInfo(traceback.format_exc())

            feedback.setProgress(int((current + 1) * 100 / total))

        # Итоговая статистика
        feedback.pushInfo('')
        feedback.pushInfo('=' * 80)
        feedback.pushInfo('📊 СТАТИСТИКА ОБРАБОТКИ:')
        feedback.pushInfo('=' * 80)
        feedback.pushInfo(f'Всего объектов в слое:     {source.featureCount()}')
        if filter_field and filter_values_str:
            feedback.pushInfo(f'После фильтрации:          {filtered_count}')
        feedback.pushInfo(f'Создано линий:             {processed_count}')
        feedback.pushInfo(f'Пропущено объектов:        {skipped_count}')
        feedback.pushInfo(f'Ошибок обработки:          {error_count}')
        feedback.pushInfo('=' * 80)

        return {self.OUTPUT: dest_id}
