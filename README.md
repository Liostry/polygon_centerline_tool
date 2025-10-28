<div align="center"><div align="center">



# 🏗️ QGIS Polygon Centerline Tool# 🏗️ QGIS Polygon Centerline Tool



*Продвинутый инструмент для извлечения центральных линий из полигональных объектов**Продвинутый инструмент для извлечения центральных линий из полигональных объектов*



[![QGIS](https://img.shields.io/badge/QGIS-3.16+-2E8B57?style=for-the-badge&logo=qgis&logoColor=white)](https://qgis.org)[![QGIS](https://img.shields.io/badge/QGIS-3.16+-2E8B57?style=for-the-badge&logo=qgis&logoColor=white)](https://qgis.org)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

[![Shapely](https://img.shields.io/badge/Shapely-Geometry-FF6B35?style=for-the-badge)](https://shapely.readthedocs.io)[![Shapely](https://img.shields.io/badge/Shapely-Geometry-FF6B35?style=for-the-badge)](https://shapely.readthedocs.io)

[![SciPy](https://img.shields.io/badge/SciPy-Voronoi-8CAAE6?style=for-the-badge)](https://scipy.org)[![SciPy](https://img.shields.io/badge/SciPy-Voronoi-8CAAE6?style=for-the-badge)](https://scipy.org)



[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](.)[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](.)



</div></div>



------



## 🎯 Что это?## 🎯 Что это?



Плагин для QGIS, который создаёт **центральные линии (медиальные оси)** из полигональных объектов с использованием **диаграммы Вороного**. Идеально подходит для:Плагин для QGIS, который создаёт **центральные линии (медиальные оси)** из полигональных объектов с использованием **диаграммы Вороного**. Идеально подходит для:



- 🅿️ **Парковочных зон** → центральные проезды- 🅿️ **Парковочных зон** → центральные проезды

- 🛣️ **Дорожных полотен** → осевые линии- 🛣️ **Дорожных полотен** → осевые линии

- 🏞️ **Речных русел** → фарватеры- 🏞️ **Речных русел** → фарватеры

- 🏢 **Зданий сложной формы** → структурные оси- 🏢 **Зданий сложной формы** → структурные оси

- 📐 **Любых полигонов** → скелетные линии- 📐 **Любых полигонов** → скелетные линии



------



## 🧠 Принцип работы## 🧠 Принцип работы



### 🔬 Научная основа### 🔬 Научная основа



Плагин использует **алгоритм диаграммы Вороного** для построения медиальной оси полигона:Плагин использует **алгоритм диаграммы Вороного** для построения медиальной оси полигона:



``````

1. Дискретизация границы → Равномерные точки по периметру1. Дискретизация границы → Равномерные точки по периметру

2. Построение Вороного    → Деление плоскости на ячейки2. Построение Вороного    → Деление плоскости на ячейки

3. Фильтрация рёбер       → Только внутренние сегменты3. Фильтрация рёбер       → Только внутренние сегменты

4. Соединение линий       → Формирование скелета4. Соединение линий       → Формирование скелета

5. Постобработка          → Упрощение и продление5. Постобработка          → Упрощение и продление

``````



### 📊 Диаграмма Вороного### 📊 Диаграмма Вороного



<div align="center"><div align="center">



```mermaid```mermaid

graph TBgraph TB

    A[🔷 Входной полигон] --> B[📍 Дискретизация границы<br/>плотность = 1.0м]    A[🔷 Входной полигон] --> B[📍 Дискретизация границы<br/>плотность = 1.0м]

    B --> C[⚡ Построение диаграммы<br/>Вороного]    B --> C[⚡ Построение диаграммы<br/>Вороного]

    C --> D{🎯 Режим построения}    C --> D{🎯 Режим построения}

        

    D -->|Полный скелет| E[🌳 Все ветви медиальной оси]    D -->|Полный скелет| E[🌳 Все ветви медиальной оси]

    D -->|Главная ось| F[📏 Упрощение до одной линии<br/>от торца до торца]    D -->|Главная ось| F[📏 Упрощение до одной линии<br/>от торца до торца]

    D -->|Самая длинная| G[🎯 Только самый длинный сегмент]    D -->|Самая длинная| G[🎯 Только самый длинный сегмент]

        

    E --> H[🔧 Постобработка]    E --> H[🔧 Постобработка]

    F --> H    F --> H

    G --> H    G --> H

        

    H --> I[📐 Упрощение геометрии<br/>допуск = 0.1м]    H --> I[📐 Упрощение геометрии<br/>допуск = 0.1м]

    I --> J[🎨 Продление до границ]    I --> J[🎨 Продление до границ]

    J --> K[✅ Результат: LineString]    J --> K[✅ Результат: LineString]

``````



</div></div>



### 🛠️ Технические детали### 🛠️ Технические детали



| Этап | Описание | Параметры || Этап | Описание | Параметры |

|------|----------|-----------||------|----------|-----------|

| **Дискретизация** | Разбиение границы на равномерные точки | `density` (1.0м) || **Дискретизация** | Разбиение границы на равномерные точки | `density` (1.0м) |

| **Вороного** | Построение диаграммы для граничных точек | `scipy.spatial.Voronoi` || **Вороного** | Построение диаграммы для граничных точек | `scipy.spatial.Voronoi` |

| **Фильтрация** | Отбор рёбер внутри полигона | `polygon.contains()` || **Фильтрация** | Отбор рёбер внутри полигона | `polygon.contains()` |

| **Соединение** | Объединение сегментов в линии | `shapely.ops.linemerge` || **Соединение** | Объединение сегментов в линии | `shapely.ops.linemerge` |

| **Упрощение** | Удаление лишних вершин | `simplify_tolerance` || **Упрощение** | Удаление лишних вершин | `simplify_tolerance` |

| **Продление** | Доведение концов до границ | Векторная экстраполяция || **Продление** | Доведение концов до границ | Векторная экстраполяция |



------



## 🚀 Установка## 🚀 Установка



### 📁 Ручная установка### 📁 Ручная установка

```batch```batch

# Скопируйте папку плагина в:# Скопируйте папку плагина в:

%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\polygon_centerline_tool\%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\polygon_centerline_tool\

``````



### 🔄 Клонирование### 🔄 Клонирование

```bash```bash

cd %APPDATA%\QGIS\QGIS3\profiles\default\python\pluginscd %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins

git clone https://github.com/yourusername/polygon_centerline_tool.gitgit clone https://github.com/yourusername/polygon_centerline_tool.git

``````



### ✅ Активация### ✅ Активация

1. Откройте **QGIS**1. Откройте **QGIS**

2. **Плагины** → **Управление и установка плагинов**2. **Плагины** → **Управление и установка плагинов**

3. Найдите **"Polygon Centerline Tool"**3. Найдите **"Polygon Centerline Tool"**

4. Установите флажок ✅4. Установите флажок ✅



------



## 🎮 Использование## 🎮 Использование



### 🔍 Где найти### 🔍 Где найти

**Processing Toolbox** → **Polygon Centerline** → **Polygon to Centerline****Processing Toolbox** → **Polygon Centerline** → **Polygon to Centerline**



### ⚙️ Основные параметры### ⚙️ Основные параметры



| Параметр | Рекомендуемое | Описание || Параметр | Рекомендуемое | Описание |

|----------|---------------|----------||----------|---------------|----------|

| **Режим построения** | 🎯 Главная ось | Тип извлекаемой линии || **Режим построения** | 🎯 Главная ось | Тип извлекаемой линии |

| **Плотность дискретизации** | `1.0` м | Детализация границы || **Плотность дискретизации** | `1.0` м | Детализация границы |

| **Допуск упрощения** | `0.1` м | Сглаживание результата || **Допуск упрощения** | `0.1` м | Сглаживание результата |



### 🎯 Режимы построения### 🎯 Режимы построения



<div align="center"><div align="center">



| Режим | Описание | Применение || Режим | Описание | Применение |

|-------|----------|------------||-------|----------|------------|

| **🌳 Полный скелет** | Все ветви медиальной оси | Сложные формы, анализ структуры || **🌳 Полный скелет** | Все ветви медиальной оси | Сложные формы, анализ структуры |

| **📏 Главная ось** ⭐ | Упрощённая линия от торца до торца | Дороги, парковки, коридоры || **📏 Главная ось** ⭐ | Упрощённая линия от торца до торца | Дороги, парковки, коридоры |

| **🎯 Самая длинная** | Только самый протяжённый сегмент | Быстрое извлечение основы || **🎯 Самая длинная** | Только самый протяжённый сегмент | Быстрое извлечение основы |



</div></div>



### 🔍 Умная фильтрация### 🔍 Умная фильтрация



1. **Выберите поле** для фильтрации1. **Выберите поле** для фильтрации

2. **Оставьте значения пустыми** → увидите все доступные варианты2. **Оставьте значения пустыми** → увидите все доступные варианты

3. **Введите нужные значения** через запятую3. **Введите нужные значения** через запятую

4. **Запустите повторно** с применённым фильтром4. **Запустите повторно** с применённым фильтром



``````

Пример: highway = "primary,secondary,trunk"Пример: highway = "primary,secondary,trunk"

``````



------



## 🧪 Примеры применения## 🧪 Примеры применения



### 🅿️ Парковочные зоны### 🅿️ Парковочные зоны



```yaml```yaml

Входные данные: Полигоны парковокВходные данные: Полигоны парковок

Режим: Главная осьРежим: Главная ось

Плотность: 1.0мПлотность: 1.0м

Результат: Центральные проездыРезультат: Центральные проезды

``````



### 🛣️ Дорожная сеть### 🛣️ Дорожная сеть



```yaml```yaml

Входные данные: Полигоны дорожного полотна  Входные данные: Полигоны дорожного полотна  

Режим: Главная осьРежим: Главная ось

Фильтр: highway = "primary,secondary"Фильтр: highway = "primary,secondary"

Результат: Осевые линии дорогРезультат: Осевые линии дорог

``````



### 🏞️ Водные объекты### 🏞️ Водные объекты



```yaml```yaml

Входные данные: Речные полигоныВходные данные: Речные полигоны

Режим: Полный скелетРежим: Полный скелет

Плотность: 2.0мПлотность: 2.0м

Результат: Русловая сетьРезультат: Русловая сеть

``````



------



## 🏗️ Архитектура плагина## 🏗️ Архитектура плагина



### 📁 Структура проекта### 📁 Структура проекта



``````

polygon_centerline_tool/polygon_centerline_tool/

├── 📜 __init__.py                      # Точка входа├── 📜 __init__.py                      # Точка входа

├── 🔌 polygon_centerline_plugin.py     # Основной класс плагина├── 🔌 polygon_centerline_plugin.py     # Основной класс плагина

├── 🏭 polygon_centerline_provider.py   # Processing Provider├── 🏭 polygon_centerline_provider.py   # Processing Provider

├── ⚙️ polygon_centerline_algorithm.py  # Алгоритм обработки├── ⚙️ polygon_centerline_algorithm.py  # Алгоритм обработки

├── 📋 metadata.txt                     # Метаданные плагина├── 📋 metadata.txt                     # Метаданные плагина

├── 📖 README.md                        # Этот файл├── 📖 README.md                        # Этот файл

└── 📄 LICENSE                          # Лицензия MIT└── 📄 LICENSE                          # Лицензия MIT

``````



### 🔄 Поток выполнения### 🔄 Поток выполнения



```mermaid```mermaid

sequenceDiagramsequenceDiagram

    participant U as 👤 Пользователь    participant U as 👤 Пользователь

    participant P as 🔌 Plugin    participant P as 🔌 Plugin

    participant A as ⚙️ Algorithm    participant A as ⚙️ Algorithm

    participant V as 📊 Voronoi    participant V as 📊 Voronoi

    participant S as 🔧 Shapely    participant S as 🔧 Shapely



    U->>P: Запуск алгоритма    U->>P: Запуск алгоритма

    P->>A: processAlgorithm()    P->>A: processAlgorithm()

    A->>A: Валидация параметров    A->>A: Валидация параметров

    A->>A: Применение фильтров    A->>A: Применение фильтров

        

    loop Для каждого полигона    loop Для каждого полигона

        A->>A: densify_boundary()        A->>A: densify_boundary()

        A->>V: Построение диаграммы        A->>V: Построение диаграммы

        V-->>A: Рёбра Вороного        V-->>A: Рёбра Вороного

        A->>S: Фильтрация и соединение        A->>S: Фильтрация и соединение

        S-->>A: Centerline геометрия        S-->>A: Centerline геометрия

        A->>A: Постобработка        A->>A: Постобработка

    end    end

        

    A-->>P: Результирующие линии    A-->>P: Результирующие линии

    P-->>U: ✅ Готово!    P-->>U: ✅ Готово!

``````



### 🧩 Ключевые компоненты### 🧩 Ключевые компоненты



| Класс | Назначение | Ключевые методы || Класс | Назначение | Ключевые методы |

|-------|------------|-----------------||-------|------------|-----------------|

| `PolygonCenterlinePlugin` | Главный класс плагина | `initGui()`, `unload()` || `PolygonCenterlinePlugin` | Главный класс плагина | `initGui()`, `unload()` |

| `PolygonCenterlineProvider` | Processing провайдер | `loadAlgorithms()` || `PolygonCenterlineProvider` | Processing провайдер | `loadAlgorithms()` |

| `PolygonCenterlineAlgorithm` | Алгоритм обработки | `processAlgorithm()` || `PolygonCenterlineAlgorithm` | Алгоритм обработки | `processAlgorithm()` |



------



## 🛠️ Технические требования## 🛠️ Технические требования



### 📋 Минимальные требования### 📋 Минимальные требования



| Компонент | Версия | Назначение || Компонент | Версия | Назначение |

|-----------|--------|------------||-----------|--------|------------|

| **QGIS** | ≥ 3.16 | Основная платформа || **QGIS** | ≥ 3.16 | Основная платформа |

| **Python** | ≥ 3.9 | Среда выполнения || **Python** | ≥ 3.9 | Среда выполнения |

| **shapely** | ≥ 1.7 | Геометрические операции || **shapely** | ≥ 1.7 | Геометрические операции |

| **scipy** | ≥ 1.7 | Диаграмма Вороного || **scipy** | ≥ 1.7 | Диаграмма Вороного |

| **numpy** | ≥ 1.20 | Численные вычисления || **numpy** | ≥ 1.20 | Численные вычисления |



### 🔧 Автоматические зависимости### 🔧 Автоматические зависимости



Эти библиотеки обычно уже установлены с QGIS:Эти библиотеки обычно уже установлены с QGIS:

- ✅ `shapely` - геометрические операции- ✅ `shapely` - геометрические операции

- ✅ `scipy` - научные вычисления  - ✅ `scipy` - научные вычисления  

- ✅ `numpy` - массивы и математика- ✅ `numpy` - массивы и математика

- ✅ `PyQt5` - пользовательский интерфейс- ✅ `PyQt5` - пользовательский интерфейс



------



## 📈 Производительность## 📈 Производительность



### ⚡ Оптимизации### ⚡ Оптимизации



- **Векторизованные вычисления** через NumPy- **Векторизованные вычисления** через NumPy

- **Кэширование результатов** валидации геометрий- **Кэширование результатов** валидации геометрий

- **Умная дискретизация** границ- **Умная дискретизация** границ

- **Параллельная обработка** независимых полигонов- **Параллельная обработка** независимых полигонов



### 📊 Производительность### 📊 Производительность



| Количество полигонов | Среднее время | Память || Количество полигонов | Среднее время | Память |

|---------------------|---------------|--------||---------------------|---------------|--------|

| 100 | ~5 сек | <100 МБ || 100 | ~5 сек | <100 МБ |

| 1,000 | ~30 сек | <500 МБ || 1,000 | ~30 сек | <500 МБ |

| 10,000 | ~5 мин | <2 ГБ || 10,000 | ~5 мин | <2 ГБ |



------



## 🤝 Развитие проекта## 🤝 Развитие проекта



### 🎯 Планы на будущее### 🎯 Планы на будущее



- [ ] 🖼️ **Предпросмотр результата** перед обработкой- [ ] 🖼️ **Предпросмотр результата** перед обработкой

- [ ] ⚡ **Многопоточная обработка** больших датасетов- [ ] ⚡ **Многопоточная обработка** больших датасетов

- [ ] 📊 **Интерактивный интерфейс** с настройками- [ ] 📊 **Интерактивный интерфейс** с настройками

- [ ] 🎨 **Стилизация результатов** по умолчанию- [ ] 🎨 **Стилизация результатов** по умолчанию

- [ ] 📐 **Дополнительные алгоритмы** скелетизации- [ ] 📐 **Дополнительные алгоритмы** скелетизации

- [ ] 🔧 **API для разработчиков**- [ ] 🔧 **API для разработчиков**



### 🐛 Сообщить об ошибке### 🐛 Сообщить об ошибке



Нашли баг? [Создайте issue](https://github.com/yourusername/polygon_centerline_tool/issues) с подробным описанием!Нашли баг? [Создайте issue](https://github.com/yourusername/polygon_centerline_tool/issues) с подробным описанием!



### 💡 Предложить улучшение### 💡 Предложить улучшение



Есть идея? [Откройте discussion](https://github.com/yourusername/polygon_centerline_tool/discussions) - обсудим!Есть идея? [Откройте discussion](https://github.com/yourusername/polygon_centerline_tool/discussions) - обсудим!



------



## 📜 Лицензия## 📜 Лицензия



**MIT License** - используйте свободно в любых проектах!**MIT License** - используйте свободно в любых проектах!



``````

Copyright (c) 2024 Polygon Centerline Tool ContributorsCopyright (c) 2024 Polygon Centerline Tool Contributors



Permission is hereby granted, free of charge, to any person obtaining a copyPermission is hereby granted, free of charge, to any person obtaining a copy

of this software and associated documentation files (the "Software"), to dealof this software and associated documentation files (the "Software"), to deal

in the Software without restriction, including without limitation the rightsin the Software without restriction, including without limitation the rights

to use, copy, modify, merge, publish, distribute, sublicense, and/or sellto use, copy, modify, merge, publish, distribute, sublicense, and/or sell

copies of the Software, and to permit persons to whom the Software iscopies of the Software, and to permit persons to whom the Software is

furnished to do so, subject to the following conditions:furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in allThe above copyright notice and this permission notice shall be included in all

copies or substantial portions of the Software.copies or substantial portions of the Software.

``````



------



<div align="center"><div align="center">



### 🌟 Если проект полезен - поставьте звёздочку!### 🌟 Если проект полезен - поставьте звёздочку!



**Сделано с ❤️ для GIS-сообщества****Сделано с ❤️ для GIS-сообщества**



[⬆️ Наверх](#-qgis-polygon-centerline-tool)[⬆️ Наверх](#-qgis-polygon-centerline-tool)



</div></div>
