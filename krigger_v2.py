import geopandas as gpd
import numpy as np
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from pathlib import Path

def simple_kriging_from_shp(
    shp_path,
    grid_step=100,
    value_col=None,
    output_tif=None,
    variogram_model='spherical',
    buffer_percent=10,
    utm_zone=None,
    hemisphere=None
):
    """
    Простой кригинг из shapefile с сохранением в GeoTIFF
    
    Parameters:
    -----------
    shp_path : str
        Путь к shapefile с точками
    grid_step : float
        Шаг сетки в метрах (размер ячейки)
    value_col : str, optional
        Название колонки со значениями. Если None, используется первая числовая колонка.
    output_tif : str, optional
        Путь для сохранения GeoTIFF. Если None, растр не сохраняется.
    variogram_model : str
        Модель вариограммы ('spherical', 'gaussian', 'exponential', 'linear')
    buffer_percent : float
        Процент буфера вокруг точек (например, 10 = 10%)
    utm_zone : int, optional
        Номер UTM зоны. Если None, определяется автоматически.
    hemisphere : str, optional
        'north' или 'south'. Если None, определяется автоматически.
    
    Returns:
    --------
    dict с ключами:
        'grid_x' : 2D array
            X координаты сетки
        'grid_y' : 2D array
            Y координаты сетки
        'z_interp' : 2D array
            Интерполированные значения
        'variance' : 2D array
            Дисперсия кригинга
        'transform' : Affine
            Геотрансформация растра
        'crs' : CRS
            Система координат
        'stats' : dict
            Статистика
    """
    
    print("="*60)
    print("ПРОСТОЙ КРИГИНГ ИЗ SHAPEFILE")
    print("="*60)
    
    # 1. ЗАГРУЗКА ТОЧЕК
    print("1. Загрузка точек...")
    gdf = gpd.read_file(shp_path)
    print(f"   Загружено точек: {len(gdf)}")
    print(f"   Исходный CRS: {gdf.crs}")
    
    # 2. ПРЕОБРАЗОВАНИЕ В UTM (МЕТРЫ)
    print("\n2. Преобразование координат...")
    if gdf.crs is None or gdf.crs.is_geographic:
        # Определяем центр данных
        center_lon = gdf.geometry.x.mean()
        center_lat = gdf.geometry.y.mean()
        
        # Определяем UTM зону, если не задана
        if utm_zone is None:
            utm_zone = int((center_lon + 180) // 6 + 1)
            print(f"   Автоматически определена UTM зона: {utm_zone}")
        
        # Определяем полушарие, если не задано
        if hemisphere is None:
            hemisphere = 'north' if center_lat >= 0 else 'south'
            print(f"   Автоматически определено полушарие: {hemisphere}")
        
        # Формируем EPSG код
        if hemisphere == 'north':
            epsg_code = 32600 + utm_zone
        else:
            epsg_code = 32700 + utm_zone
        
        target_crs = f'EPSG:{epsg_code}'
        print(f"   Преобразование в: {target_crs}")
        gdf = gdf.to_crs(target_crs)
    else:
        print(f"   CRS уже в метрах: {gdf.crs}")
        target_crs = gdf.crs
    
    # 3. ИЗВЛЕЧЕНИЕ ДАННЫХ
    print("\n3. Извлечение данных...")
    x = gdf.geometry.x.values.astype(np.float64)
    y = gdf.geometry.y.values.astype(np.float64)
    
    # Определяем колонку со значениями
    if value_col is None:
        # Ищем первую числовую колонку
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value_col = numeric_cols[0]
            print(f"   Используется колонка: '{value_col}'")
        else:
            raise ValueError("Не найдены числовые колонки. Укажите value_col.")
    
    z = gdf[value_col].values.astype(np.float64)
    
    # Удаляем NaN значения
    mask = ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]
    
    print(f"   Валидных точек: {len(z)}")
    print(f"   Min: {z.min():.4f}, Max: {z.max():.4f}, Mean: {z.mean():.4f}")
    
    # 4. СОЗДАНИЕ СЕТКИ
    print("\n4. Создание сетки...")
    
    # Определяем границы
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Добавляем буфер
    x_buffer = (x_max - x_min) * (buffer_percent / 100)
    y_buffer = (y_max - y_min) * (buffer_percent / 100)
    
    x_min -= x_buffer
    x_max += x_buffer
    y_min -= y_buffer
    y_max += y_buffer
    
    # Создаем регулярную сетку
    gridx = np.arange(x_min, x_max, grid_step)
    gridy = np.arange(y_min, y_max, grid_step)
    
    print(f"   Границы с буфером {buffer_percent}%:")
    print(f"     X: {x_min:.1f} - {x_max:.1f} м")
    print(f"     Y: {y_min:.1f} - {y_max:.1f} м")
    print(f"   Шаг сетки: {grid_step} м")
    print(f"   Размер сетки: {len(gridx)} x {len(gridy)} = {len(gridx)*len(gridy):,} ячеек")
    
    # 5. ВЫПОЛНЕНИЕ КРИГИНГА
    print("\n5. Выполнение кригинга...")
    
    try:
        OK = OrdinaryKriging(
            x, y, z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
            coordinates_type='euclidean',
            nlags=20,
            weight=True
        )
        
        z_interp, variance = OK.execute('grid', gridx, gridy)
        z_interp = z_interp.T  # транспонируем для (rows, cols)
        variance = variance.T
        
        print(f"   Кригинг успешно завершен!")
        print(f"   Модель вариограммы: {variogram_model}")
        
    except Exception as e:
        print(f"   Ошибка при кригинге: {e}")
        print("   Использую IDW как запасной вариант...")
        z_interp, variance = inverse_distance_weighting(x, y, z, gridx, gridy)
    
    # 6. СОХРАНЕНИЕ В GEOTIFF
    if output_tif:
        print(f"\n6. Сохранение в GeoTIFF: {output_tif}")
        save_geotiff(output_tif, z_interp, variance, x_min, y_max, grid_step, target_crs)
    
    # 7. ПОДГОТОВКА РЕЗУЛЬТАТОВ
    # Создаем 2D массивы координат
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    
    # Статистика
    stats = {
        'n_points': len(z),
        'min_value': float(z.min()),
        'max_value': float(z.max()),
        'mean_value': float(z.mean()),
        'std_value': float(z.std()),
        'grid_shape': z_interp.shape,
        'cell_size': grid_step,
        'bounds': (float(x_min), float(x_max), float(y_min), float(y_max)),
        'variogram_model': variogram_model,
        'crs': str(target_crs)
    }
    
    print("\n" + "="*60)
    print("КРИГИНГ ЗАВЕРШЕН!")
    print("="*60)
    
    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'z_interp': z_interp,
        'variance': variance,
        'transform': from_origin(x_min, y_max, grid_step, grid_step),
        'crs': target_crs,
        'stats': stats,
        'original_points': {'x': x, 'y': y, 'z': z}
    }


def inverse_distance_weighting(x, y, z, gridx, gridy, power=2, radius=None):
    """
    Простая интерполяция методом обратных расстояний (IDW)
    как запасной вариант если кригинг не работает
    """
    from scipy.spatial import cKDTree
    
    # Создаем сетку
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Точки данных
    data_points = np.column_stack([x, y])
    
    # KD-дерево для быстрого поиска
    tree = cKDTree(data_points)
    
    # Определяем радиус поиска
    if radius is None:
        # Среднее расстояние до ближайшего соседа * 3
        distances, _ = tree.query(data_points, k=2)
        radius = np.mean(distances[:, 1]) * 3
    
    # Ищем соседей в радиусе
    indices = tree.query_ball_point(grid_points, radius)
    
    # Интерполяция
    z_interp_flat = np.zeros(len(grid_points))
    variance_flat = np.zeros(len(grid_points))
    
    for i, idx_list in enumerate(indices):
        if len(idx_list) > 0:
            # Расстояния до соседей
            distances = np.sqrt(
                (grid_points[i, 0] - x[idx_list])**2 + 
                (grid_points[i, 1] - y[idx_list])**2
            )
            
            # Веса (обратное расстояние в степени power)
            weights = 1.0 / (distances**power + 1e-10)
            weights = weights / weights.sum()
            
            # Взвешенное среднее
            z_interp_flat[i] = np.sum(weights * z[idx_list])
            variance_flat[i] = np.var(z[idx_list])
        else:
            # Если нет соседей, используем среднее по всем точкам
            z_interp_flat[i] = np.mean(z)
            variance_flat[i] = np.var(z)
    
    # Преобразуем обратно в 2D
    z_interp = z_interp_flat.reshape(len(gridy), len(gridx))
    variance = variance_flat.reshape(len(gridy), len(gridx))
    
    return z_interp, variance


def save_geotiff(output_path, z_interp, variance, x_min, y_max, grid_step, crs):
    """
    Сохраняет результат в GeoTIFF
    """
    # Создаем директорию если нужно
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Определяем геотрансформацию
    # x_min, y_max - координаты верхнего левого угла
    transform = from_origin(x_min, y_max, grid_step, -grid_step)
    
    # Метаданные
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999,
        'width': z_interp.shape[1],
        'height': z_interp.shape[0],
        'count': 2,  # два канала: значения и дисперсия
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',  # сжатие для экономии места
        'tiled': True,      # тайлирование для быстрого доступа
        'blockxsize': 256,
        'blockysize': 256
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        # Записываем интерполированные значения (канал 1)
        dst.write(z_interp.astype(np.float32), 1)
        
        # Записываем дисперсию (канал 2)
        dst.write(variance.astype(np.float32), 2)
        
        # Добавляем описание каналов
        dst.set_band_description(1, "Interpolated values")
        dst.set_band_description(2, "Kriging variance")
    
    print(f"   Файл сохранен: {output_path}")
    print(f"   Размер: {z_interp.shape[1]} x {z_interp.shape[0]}")
    print(f"   Канал 1: Интерполированные значения")
    print(f"   Канал 2: Дисперсия кригинга")


def visualize_kriging_simple(result, save_figure=None):
    """
    Простая визуализация результата кригинга
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Интерполированные значения
    ax1 = axes[0]
    im1 = ax1.imshow(result['z_interp'], 
                    extent=[result['grid_x'].min(), result['grid_x'].max(),
                            result['grid_y'].min(), result['grid_y'].max()],
                    origin='lower', cmap='viridis', aspect='auto')
    ax1.scatter(result['original_points']['x'],
                result['original_points']['y'],
                c='red', s=20, alpha=0.6, edgecolors='white')
    ax1.set_title('Результат кригинга')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    plt.colorbar(im1, ax=ax1, label='Значение')
    
    # 2. Дисперсия
    ax2 = axes[1]
    im2 = ax2.imshow(result['variance'],
                    extent=[result['grid_x'].min(), result['grid_x'].max(),
                            result['grid_y'].min(), result['grid_y'].max()],
                    origin='lower', cmap='hot', aspect='auto')
    ax2.set_title('Дисперсия кригинга')
    ax2.set_xlabel('X (м)')
    ax2.set_ylabel('Y (м)')
    plt.colorbar(im2, ax=ax2, label='Дисперсия')
    
    # Информация
    stats = result['stats']
    info_text = (f"Точек: {stats['n_points']}\n"
                 f"Сетка: {stats['grid_shape'][1]}×{stats['grid_shape'][0]}\n"
                 f"Ячейка: {stats['cell_size']} м\n"
                 f"CRS: {stats['crs']}")
    
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle('Кригинг: интерполяция и дисперсия', fontsize=14)
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(save_figure, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_figure}")
    
    plt.show()
    
    # Вывод статистики
    print("\n" + "="*60)
    print("СТАТИСТИКА РЕЗУЛЬТАТА:")
    print("="*60)
    print(f"Точек: {stats['n_points']}")
    print(f"Размер растра: {stats['grid_shape'][1]} x {stats['grid_shape'][0]}")
    print(f"Шаг сетки: {stats['cell_size']} м")
    print(f"Границы: X({stats['bounds'][0]:.1f}, {stats['bounds'][1]:.1f}) м")
    print(f"          Y({stats['bounds'][2]:.1f}, {stats['bounds'][3]:.1f}) м")
    print(f"Интерполированные значения: {np.nanmin(result['z_interp']):.4f} - "
          f"{np.nanmax(result['z_interp']):.4f}")
    print(f"Средняя дисперсия: {np.nanmean(result['variance']):.4f}")


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Пример 1: Простой вызов с сохранением
    result = simple_kriging_from_shp(
        shp_path=r"D:\ml_datasets\Забайкалье\archive\FINAL_ORE_DEPOSITS_AU_AG_CU_MO_porph_ZAB_AMU_Makarov_2025-12-10_EPSG4326_AOI_67.zip",
        grid_step=100,  # 100 метров
        # value_col="grade",  # колонка со значениями
        output_tif="kriging_result.tif",  # сохранить в GeoTIFF
        variogram_model='spherical',
        buffer_percent=10
    )
    
    # Визуализация
    visualize_kriging_simple(result, save_figure="kriging_plot.png")
    
    # # Пример 2: С указанием UTM зоны
    # result2 = simple_kriging_from_shp(
    #     shp_path="точки_забайкалье.shp",
    #     grid_step=50,  # 50 метров - более детально
    #     value_col="cu_content",
    #     output_tif="забайкалье_кригинг.tif",
    #     utm_zone=50,  # UTM зона 50 (для Забайкалья)
    #     hemisphere='north',
    #     variogram_model='gaussian'
    # )
    
    # Пример 3: Быстрое создание карты с большим шагом
    # result3 = simple_kriging_from_shp(
    #     shp_path="данные.shp",
    #     grid_step=500,  # 500 метров - для обзора
    #     output_tif="обзорная_карта.tif",
    #     buffer_percent=20  # 20% буфер
    # )