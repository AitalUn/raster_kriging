import rasterio
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_point_indices_in_raster(raster_path, shp_path, value_column=None, nodata_value=-9999):
    """
    Загружает растр и точки, возвращает индексы строк и столбцов точек в numpy массиве растра.
    
    Parameters:
    -----------
    raster_path : str
        Путь к растровому файлу (GeoTIFF)
    shp_path : str
        Путь к shapefile с точками
    value_column : str, optional
        Имя столбца с значениями в shapefile (если нужны атрибуты точек)
    nodata_value : float
        Значение, интерпретируемое как NoData в растре
        
    Returns:
    --------
    dict с ключами:
        'rows' : np.array
            Индексы строк точек в растре (ось Y)
        'cols' : np.array
            Индексы столбцов точек в растре (ось X)
        'values' : np.array
            Значения растра в этих точках
        'coords' : np.array
            Географические координаты точек (x, y)
        'point_attributes' : np.array или None
            Атрибуты точек из shapefile, если указан value_column
        'raster_shape' : tuple
            Размеры растра (rows, cols)
        'transform' : Affine
            Гео-трансформация растра
        'crs' : CRS
            Система координат растра
        'bounds' : BoundingBox
            Границы растра
    """
    
    print("="*60)
    print("ЗАГРУЗКА РАСТРА И ПОЛУЧЕНИЕ ИНДЕКСОВ ТОЧЕК")
    print("="*60)
    
    results = {}
    
    # 1. ЗАГРУЗКА РАСТРА
    print("\n1. Загрузка растра...")
    with rasterio.open(raster_path) as src:
        # Читаем данные растра
        raster_data = src.read(1)  # первый канал
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
        # Сохраняем метаданные
        results['raster_shape'] = (src.height, src.width)
        results['transform'] = transform
        results['crs'] = crs
        results['bounds'] = bounds
        results['raster_data'] = raster_data  # полный массив растра
        
        print(f"   Размер растра: {src.width} x {src.height}")
        print(f"   CRS: {crs}")
        print(f"   Границы: {bounds}")
        print(f"   Размер ячейки: {transform.a:.2f} x {abs(transform.e):.2f}")
    
    # 2. ЗАГРУЗКА ТОЧЕК
    print("\n2. Загрузка точек...")
    gdf = gpd.read_file(shp_path)
    print(f"   Загружено точек: {len(gdf)}")
    print(f"   CRS точек: {gdf.crs}")
    
    # 3. ПРЕОБРАЗОВАНИЕ КООРДИНАТ
    print("\n3. Преобразование координат...")
    if gdf.crs != crs:
        print(f"   Преобразование CRS: {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)
    
    # Извлекаем координаты точек
    points_x = gdf.geometry.x.values
    points_y = gdf.geometry.y.values
    
    # 4. ПОЛУЧЕНИЕ ИНДЕКСОВ И ЗНАЧЕНИЙ
    print("\n4. Получение индексов точек в растре...")
    
    # Подготовка массивов
    rows_list = []      # индексы строк (ось Y)
    cols_list = []      # индексы столбцов (ось X)
    values_list = []    # значения из растра
    coords_list = []    # географические координаты
    attr_list = [] if value_column else None  # атрибуты точек
    
    # Счетчики для статистики
    total_points = len(gdf)
    points_inside = 0
    points_valid_data = 0
    
    # Открываем растр снова для доступа к методу index()
    with rasterio.open(raster_path) as src:
        for i, (x, y) in enumerate(zip(points_x, points_y)):
            # Проверяем, находится ли точка внутри границ растра
            if (bounds.left <= x <= bounds.right and 
                bounds.bottom <= y <= bounds.top):
                
                points_inside += 1
                
                # Получаем индексы пикселя
                row, col = src.index(x, y)  # <- КЛЮЧЕВАЯ СТРОКА!
                
                # Проверяем, что индексы в пределах массива
                if (0 <= row < src.height and 0 <= col < src.width):
                    # Получаем значение из растра
                    value = raster_data[row, col]
                    
                    # Проверяем, не является ли значение NoData
                    is_nodata = (
                        np.isnan(value) or 
                        value <= nodata_value or 
                        value >= 3.4028235e+38
                    )
                    
                    if not is_nodata:
                        points_valid_data += 1
                        
                        # Сохраняем данные
                        rows_list.append(row)
                        cols_list.append(col)
                        values_list.append(value)
                        coords_list.append([x, y])
                        
                        # Сохраняем атрибут точки, если указан
                        if value_column and value_column in gdf.columns:
                            attr_list.append(gdf.iloc[i][value_column])
    
    # 5. ПРЕОБРАЗОВАНИЕ В NUMPY МАССИВЫ
    print("\n5. Формирование результатов...")
    
    results['rows'] = np.array(rows_list, dtype=np.int32)
    results['cols'] = np.array(cols_list, dtype=np.int32)
    results['values'] = np.array(values_list, dtype=np.float32)
    results['coords'] = np.array(coords_list, dtype=np.float64)
    
    if value_column and attr_list:
        results['point_attributes'] = np.array(attr_list)
        print(f"   Атрибуты точек загружены из колонки: '{value_column}'")
    else:
        results['point_attributes'] = None
    
    # 6. ВЫВОД СТАТИСТИКИ
    print("\n" + "="*60)
    print("СТАТИСТИКА:")
    print("="*60)
    print(f"Всего точек в shapefile: {total_points}")
    print(f"Точек внутри границ растра: {points_inside} ({points_inside/total_points*100:.1f}%)")
    print(f"Точек с валидными данными: {points_valid_data} ({points_valid_data/total_points*100:.1f}%)")
    
    if points_valid_data > 0:
        print(f"\nИНДЕКСЫ (первые 5 точек):")
        print(f"{'№':<4} {'Row':<6} {'Col':<6} {'Value':<10} {'X':<12} {'Y':<12}")
        print("-" * 50)
        for i in range(min(5, len(results['rows']))):
            print(f"{i:<4} {results['rows'][i]:<6} {results['cols'][i]:<6} "
                  f"{results['values'][i]:<10.4f} "
                  f"{results['coords'][i][0]:<12.2f} {results['coords'][i][1]:<12.2f}")
        
        print(f"\nСтатистика значений растра в точках:")
        print(f"  Минимум: {results['values'].min():.4f}")
        print(f"  Максимум: {results['values'].max():.4f}")
        print(f"  Среднее: {results['values'].mean():.4f}")
        print(f"  Стандартное отклонение: {results['values'].std():.4f}")
    
    print("="*60)
    
    return results


# ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ДЛЯ РАБОТЫ С ИНДЕКСАМИ
def visualize_points_on_raster(results, max_points_to_show=1000):
    """
    Визуализирует точки на растре
    """
    import matplotlib.pyplot as plt
    
    raster = results['raster_data']
    rows = results['rows']
    cols = results['cols']
    values = results['values']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Точки на растре
    ax1 = axes[0]
    im1 = ax1.imshow(raster, cmap='viridis', 
                    vmin=np.percentile(raster[raster > -9999], 2),
                    vmax=np.percentile(raster[raster > -9999], 98))
    
    # Ограничиваем количество отображаемых точек для скорости
    if len(rows) > max_points_to_show:
        idx = np.random.choice(len(rows), max_points_to_show, replace=False)
        display_rows = rows[idx]
        display_cols = cols[idx]
        display_values = values[idx]
    else:
        display_rows = rows
        display_cols = cols
        display_values = values
    
    scatter1 = ax1.scatter(display_cols, display_rows, 
                          c=display_values, cmap='hot', 
                          s=50, alpha=0.7, edgecolors='white')
    ax1.set_title(f'Точки на растре (показано {len(display_rows)} из {len(rows)})')
    ax1.set_xlabel('Column (X)')
    ax1.set_ylabel('Row (Y)')
    plt.colorbar(im1, ax=ax1, label='Значение растра')
    
    # 2. Маска точек
    ax2 = axes[1]
    point_mask = np.zeros_like(raster, dtype=bool)
    point_mask[rows, cols] = True
    
    im2 = ax2.imshow(point_mask, cmap='Reds')
    ax2.set_title('Бинарная маска точек')
    ax2.set_xlabel('Column (X)')
    ax2.set_ylabel('Row (Y)')
    
    # Подписываем количество точек
    ax2.text(0.02, 0.98, f'Точек: {len(rows)}', 
             transform=ax2.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def extract_point_neighborhoods(results, window_size=3):
    """
    Извлекает окрестности вокруг каждой точки
    """
    raster = results['raster_data']
    rows = results['rows']
    cols = results['cols']
    
    neighborhoods = []
    statistics = []
    
    half = window_size // 2
    
    for i, (row, col) in enumerate(zip(rows, cols)):
        # Определяем границы окна
        row_start = max(0, row - half)
        row_end = min(raster.shape[0], row + half + 1)
        col_start = max(0, col - half)
        col_end = min(raster.shape[1], col + half + 1)
        
        # Извлекаем окно
        window = raster[row_start:row_end, col_start:col_end]
        neighborhoods.append(window)
        
        # Собираем статистику
        if window.size > 0:
            stats = {
                'point_id': i,
                'row': row,
                'col': col,
                'window_shape': window.shape,
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'center_value': raster[row, col]  # значение в самой точке
            }
            statistics.append(stats)
    
    print(f"\nИзвлечено {len(neighborhoods)} окрестностей размером ~{window_size}x{window_size}")
    
    return neighborhoods, statistics

def create_point_density_raster(results, buffer_radius=3):
    """
    Создает растр плотности точек (буферы вокруг точек)
    """
    from scipy.ndimage import binary_dilation
    
    raster_shape = results['raster_shape']
    
    # Создаем бинарную маску точек
    point_mask = np.zeros(raster_shape, dtype=bool)
    point_mask[results['rows'], results['cols']] = True
    
    # Создаем буферы вокруг точек
    if buffer_radius > 0:
        # Создаем структурирующий элемент для дилатации
        structure = np.ones((buffer_radius*2+1, buffer_radius*2+1), dtype=bool)
        density_mask = binary_dilation(point_mask, structure=structure)
    else:
        density_mask = point_mask
    
    # Преобразуем в плотность (расстояние до ближайшей точки)
    from scipy.ndimage import distance_transform_edt
    if np.any(point_mask):
        distance_to_points = distance_transform_edt(~point_mask)
        # Инвертируем: чем ближе к точке, тем выше значение
        density_raster = 1.0 / (1.0 + distance_to_points)
    else:
        density_raster = np.zeros(raster_shape)
    
    return {
        'point_mask': point_mask.astype(np.uint8),
        'density_mask': density_mask.astype(np.uint8),
        'density_raster': density_raster,
        'distance_to_points': distance_to_points if 'distance_to_points' in locals() else None
    }


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Основное использование
    results = get_point_indices_in_raster(
        raster_path="landsat_channel.tif",
        shp_path="рудопроявления.shp",
        value_column="grade"  # опционально: если есть атрибут в точках
    )
    
    print("\n" + "="*60)
    print("ДОСТУП К ДАННЫМ:")
    print("="*60)
    
    # Прямой доступ к индексам
    print(f"\n1. Индексы строк (первые 10): {results['rows'][:10]}")
    print(f"2. Индексы столбцов (первые 10): {results['cols'][:10]}")
    
    # Проверка: получаем значения по индексам
    print("\n3. Проверка соответствия индексов и значений:")
    for i in range(min(3, len(results['rows']))):
        row, col = results['rows'][i], results['cols'][i]
        value_from_indices = results['raster_data'][row, col]
        value_stored = results['values'][i]
        print(f"   Точка {i}: row={row}, col={col}, "
              f"значение из индексов={value_from_indices:.4f}, "
              f"сохраненное значение={value_stored:.4f}, "
              f"совпадение: {np.isclose(value_from_indices, value_stored)}")
    
    # Визуализация
    print("\n4. Визуализация...")
    fig = visualize_points_on_raster(results)
    
    # Извлечение окрестностей
    print("\n5. Извлечение окрестностей точек...")
    neighborhoods, stats = extract_point_neighborhoods(results, window_size=5)
    
    # Создание растра плотности
    print("\n6. Создание растра плотности точек...")
    density_results = create_point_density_raster(results, buffer_radius=2)
    
    print("\n" + "="*60)
    print("ВСЕ ГОТОВО! Вы можете использовать results['rows'] и results['cols']")
    print("для доступа к точкам в numpy массиве растра.")
    print("="*60)