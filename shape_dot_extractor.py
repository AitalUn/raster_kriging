import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
# from pykrige.ok import OrdinaryKriging
from rasterio.plot import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_raster_and_points(raster_path, shp_path):
    """
    Загружает растр и точки, находит значения растра в точках
    """
    # 1. Загружаем растр
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # первый канал
        raster_profile = src.profile
        raster_crs = src.crs
        
        # Получаем гео-трансформацию
        transform = src.transform
        bounds = src.bounds
        
        print(f"Растр: {src.width}x{src.height} пикселей")
        print(f"CRS растра: {raster_crs}")
        print(f"Границы: {bounds}")
        
        # Координаты углов растра
        x_coords = np.linspace(bounds.left, bounds.right, src.width)
        y_coords = np.linspace(bounds.bottom, bounds.top, src.height)
        
        # Сетка для интерполяции
        gridx, gridy = np.meshgrid(x_coords, y_coords)
    
    # 2. Загружаем точки
    gdf = gpd.read_file(shp_path)
    print(f"Загружено {len(gdf)} точек")
    print(f"CRS точек: {gdf.crs}")
    
    # 3. Приводим к одной системе координат
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        print(f"Точки преобразованы в CRS растра: {gdf.crs}")
    
    # 4. Извлекаем координаты точек
    points_x = gdf.geometry.x.values
    points_y = gdf.geometry.y.values
    
    # 5. Находим значения растра в точках
    print("\nИзвлечение значений растра в точках...")
    raster_values = []
    valid_indices = []
    
    with rasterio.open(raster_path) as src:
        for i, (x, y) in enumerate(zip(points_x, points_y)):
            # Проверяем, попадает ли точка в границы растра
            if (bounds.left <= x <= bounds.right and 
                bounds.bottom <= y <= bounds.top):
                
                # Преобразуем координаты в индексы пикселей
                row, col = src.index(x, y)
                
                # Проверяем, что индексы в пределах массива
                if (0 <= row < src.height and 0 <= col < src.width):
                    value = raster_data[row, col]
                    
                    # Проверяем на NoData (обычно это очень большое или маленькое число)
                    if not (np.isnan(value) or value <= -9999 or value >= 3.4028235e+38):
                        raster_values.append(value)
                        valid_indices.append(i)
    
    print(f"Найдено значений для {len(raster_values)} из {len(points_x)} точек")
    
    # Фильтруем точки с валидными значениями
    valid_x = points_x[valid_indices]
    valid_y = points_y[valid_indices]
    raster_values = np.array(raster_values)
    
    return {
        'raster_data': raster_data,
        'raster_profile': raster_profile,
        'raster_crs': raster_crs,
        'transform': transform,
        'bounds': bounds,
        'gridx': gridx,
        'gridy': gridy,
        'points_x': valid_x,
        'points_y': valid_y,
        'raster_values': raster_values,
        'gdf': gdf.iloc[valid_indices] if len(valid_indices) > 0 else gdf
    }

