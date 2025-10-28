# polygon_centerline_tool
QGIS plugin for creating centerlines from polygon features
![QGIS](https://img.shields.io/badge/QGIS-3.16+-green?logo=qgis)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

```markdown
# QGIS Polygon to Centerline

Плагин для создания центральных линий из полигонов методом Voronoi.

## Установка

Скопируйте папку в:
```
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\polygon_centerline_tool\
```

Или клонируйте репозиторий:
```
cd %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins
git clone https://github.com/ваш_username/polygon_centerline_tool.git
```

## Использование

**Processing Toolbox** → **Polygon Centerline** → **Polygon to Centerline**

### Параметры

- **Режим**: Главная ось (от торца до торца) — рекомендуется
- **Плотность**: 1.0 м
- **Упрощение**: 0.1 м

### Фильтрация

1. Выберите поле и оставьте значения пустыми → увидите доступные значения
2. Введите значения через запятую и запустите снова

## Требования

- QGIS >= 3.16
- scipy, shapely (обычно уже есть)

## Лицензия

MIT
```
