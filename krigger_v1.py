import numpy as np
import matplotlib.pyplot as plt
import rasterio as rs
from dot_extractor_v2 import get_point_indices_in_raster
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

def perform_kriging_interpolation(result, raster_shape, output_path=None, 
                                  variogram_model='spherical', cell_size=None):
    """
    Выполняет кригинг для интерполяции значений точек на регулярную сетку.
    
    Parameters:
    -----------
    result : dict
        Результат функции get_point_indices_in_raster
    raster_shape : tuple
        Размер выходного растра (rows, cols)
    output_path : str, optional
        Путь для сохранения результата
    variogram_model : str
        Модель вариограммы ('spherical', 'gaussian', 'exponential', 'linear')
    cell_size : float, optional
        Размер ячейки в метрах (если None, используется размер исходного растра)
    
    Returns:
    --------
    kriged_raster : np.array
        Интерполированный растр
    """
    
    print("="*60)
    print("ГЕНЕРАЦИЯ РАСТРА МЕТОДОМ КРИГИНГА")
    print("="*60)
    
    # 1. ПОДГОТОВКА ДАННЫХ
    print("\n1. Подготовка данных для кригинга...")
    
    # Извлекаем координаты точек и значения
    x_coords = result['coords'][:, 0]  # долгота или X
    y_coords = result['coords'][:, 1]  # широта или Y
    z_values = result['values']        # значения в точках
    
    print(f"   Количество точек: {len(x_coords)}")
    print(f"   Диапазон значений: {z_values.min():.4f} - {z_values.max():.4f}")
    print(f"   Среднее значение: {z_values.mean():.4f}")
    
    # 2. СОЗДАНИЕ СЕТКИ ДЛЯ ИНТЕРПОЛЯЦИИ
    print("\n2. Создание сетки интерполяции...")
    
    # Определяем границы сетки
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Добавляем буфер вокруг точек (10%)
    x_buffer = (x_max - x_min) * 0.1
    y_buffer = (y_max - y_min) * 0.1
    
    x_min -= x_buffer
    x_max += x_buffer
    y_min -= y_buffer
    y_max += y_buffer
    
    # Создаем регулярную сетку
    if cell_size is None:
        # Автоматически определяем размер ячейки на основе расстояний между точками
        from scipy.spatial import distance
        if len(x_coords) > 100:
            # Берем подвыборку для ускорения
            idx = np.random.choice(len(x_coords), 100, replace=False)
            sample_x = x_coords[idx]
            sample_y = y_coords[idx]
        else:
            sample_x = x_coords
            sample_y = y_coords
        
        # Вычисляем среднее расстояние до ближайшего соседа
        points = np.column_stack([sample_x, sample_y])
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2: сама точка и ближайший сосед
        mean_distance = np.mean(distances[:, 1])  # исключаем нулевое расстояние
        
        # Размер ячейки = 1/3 среднего расстояния
        cell_size = mean_distance / 3
        print(f"   Автоматический размер ячейки: {cell_size:.2f} м")
    
    # Создаем координаты сетки
    gridx = np.arange(x_min, x_max, cell_size)
    gridy = np.arange(y_min, y_max, cell_size)
    
    # Создаем сетку для кригинга
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    
    print(f"   Размер сетки: {len(gridx)} x {len(gridy)} = {len(gridx)*len(gridy)} ячеек")
    print(f"   Размер ячейки: {cell_size:.2f} м")
    
    # 3. ВЫПОЛНЕНИЕ КРИГИНГА
    print("\n3. Выполнение кригинга...")
    
    try:
        # Создаем модель обычного кригинга
        OK = OrdinaryKriging(
            x_coords, y_coords, z_values,
            variogram_model=variogram_model,
            verbose=True,
            enable_plotting=True,  # покажет график вариограммы
            coordinates_type='euclidean',
            nlags=20,
            weight=True
        )
        
        # Выполняем интерполяцию на сетке
        print("   Выполняется интерполяция... (это может занять время)")
        z_interp, ss = OK.execute('grid', gridx, gridy)
        
        print(f"   Кригинг выполнен успешно!")
        print(f"   Модель вариограммы: {variogram_model}")
        print(f"   Параметры вариограммы:")
        print(f"     Nugget: {OK.variogram_model_parameters[0]:.4f}")
        print(f"     Sill: {OK.variogram_model_parameters[1]:.4f}")
        print(f"     Range: {OK.variogram_model_parameters[2]:.4f}")
        
    except Exception as e:
        print(f"   Ошибка при кригинге: {e}")
        print("   Использую простую интерполяцию (griddata) как запасной вариант...")
        
        # Запасной вариант: линейная интерполяция
        z_interp = griddata(
            (x_coords, y_coords), z_values,
            (grid_x, grid_y), method='linear',
            fill_value=np.nanmean(z_values)
        )
        ss = None
    
    # 4. ПРЕОБРАЗОВАНИЕ К ИСХОДНОМУ РАЗМЕРУ
    print("\n4. Преобразование к исходному размеру...")
    
    # Если нужен растр определенного размера (например, как исходный)
    target_rows, target_cols = raster_shape
    
    # Создаем целевую сетку координат
    target_x = np.linspace(x_min, x_max, target_cols)
    target_y = np.linspace(y_min, y_max, target_rows)
    target_grid_x, target_grid_y = np.meshgrid(target_x, target_y)
    
    # Интерполируем на целевую сетку
    if ss is not None:
        # Для кригинга интерполируем и значения, и дисперсию
        kriged_values = griddata(
            (grid_x.flatten(), grid_y.flatten()), z_interp.flatten(),
            (target_grid_x, target_grid_y), method='linear',
            fill_value=np.nanmean(z_interp)
        )
        
        kriged_variance = griddata(
            (grid_x.flatten(), grid_y.flatten()), ss.flatten(),
            (target_grid_x, target_grid_y), method='linear',
            fill_value=np.nanmean(ss)
        )
    else:
        # Для простой интерполяции
        kriged_values = griddata(
            (grid_x.flatten(), grid_y.flatten()), z_interp.flatten(),
            (target_grid_x, target_grid_y), method='linear',
            fill_value=np.nanmean(z_interp)
        )
        kriged_variance = None
    
    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТА
    if output_path:
        print(f"\n5. Сохранение результата в {output_path}...")
        
        # Используем метаданные из исходного растра
        with rs.open(raster_path) as src:
            meta = src.meta.copy()
        
        # Обновляем метаданные
        meta.update({
            'dtype': 'float32',
            'count': 2 if kriged_variance is not None else 1,
            'height': target_rows,
            'width': target_cols,
            'transform': rs.transform.from_bounds(
                x_min, y_min, x_max, y_max, 
                target_cols, target_rows
            )
        })
        
        # Сохраняем
        with rs.open(output_path, 'w', **meta) as dst:
            dst.write(kriged_values.astype(np.float32), 1)
            if kriged_variance is not None:
                dst.write(kriged_variance.astype(np.float32), 2)
        
        print(f"   Растр успешно сохранен")
    
    print("\n" + "="*60)
    print("КРИГИНГ ЗАВЕРШЕН!")
    print("="*60)
    
    return {
        'kriged_raster': kriged_values,
        'kriged_variance': kriged_variance,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'z_interp': z_interp,
        'parameters': {
            'variogram_model': variogram_model,
            'cell_size': cell_size,
            'grid_shape': (len(gridx), len(gridy))
        }
    }


def compare_kriging_with_original(original_raster, kriged_result, result, 
                                  downsample_factor=10):
    """
    Сравнивает исходный растр и результат кригинга
    """
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    # Уменьшаем размер для визуализации
    original_down = original_raster[::downsample_factor, ::downsample_factor]
    kriged_down = kriged_result['kriged_raster'][::downsample_factor, ::downsample_factor]
    
    # Координаты точек для отображения
    points_x = result['cols'] // downsample_factor
    points_y = result['rows'] // downsample_factor
    
    # Создаем фигуру
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Исходный растр
    ax1 = axes[0, 0]
    im1 = ax1.imshow(original_down, cmap='viridis',
                    vmin=np.percentile(original_down[original_down > -9999], 2),
                    vmax=np.percentile(original_down[original_down > -9999], 98))
    ax1.scatter(points_x, points_y, c='red', s=10, alpha=0.6, edgecolors='white')
    ax1.set_title('Исходный растр с точками')
    plt.colorbar(im1, ax=ax1, label='Значение')
    
    # 2. Результат кригинга
    ax2 = axes[0, 1]
    im2 = ax2.imshow(kriged_down, cmap='viridis',
                    vmin=np.percentile(kriged_down[~np.isnan(kriged_down)], 2),
                    vmax=np.percentile(kriged_down[~np.isnan(kriged_down)], 98))
    ax2.scatter(points_x, points_y, c='red', s=10, alpha=0.6, edgecolors='white')
    ax2.set_title('Результат кригинга')
    plt.colorbar(im2, ax=ax2, label='Значение')
    
    # 3. Разница
    ax3 = axes[0, 2]
    # Масштабируем к одинаковому размеру для сравнения
    min_size = min(original_down.shape[0], kriged_down.shape[0],
                   original_down.shape[1], kriged_down.shape[1])
    
    orig_cropped = original_down[:min_size, :min_size]
    krig_cropped = kriged_down[:min_size, :min_size]
    
    diff = orig_cropped - krig_cropped
    diff_valid = diff[~np.isnan(diff)]
    
    im3 = ax3.imshow(diff, cmap='RdBu_r',
                    vmin=np.percentile(diff_valid, 5),
                    vmax=np.percentile(diff_valid, 95))
    ax3.set_title(f'Разница (MAE: {np.mean(np.abs(diff_valid)):.4f})')
    plt.colorbar(im3, ax=ax3, label='Разница')
    
    # 4. Дисперсия кригинга (если есть)
    ax4 = axes[1, 0]
    if kriged_result['kriged_variance'] is not None:
        variance_down = kriged_result['kriged_variance'][::downsample_factor, ::downsample_factor]
        im4 = ax4.imshow(variance_down[:min_size, :min_size], cmap='hot')
        ax4.set_title('Дисперсия кригинга')
        plt.colorbar(im4, ax=ax4, label='Дисперсия')
    else:
        ax4.text(0.5, 0.5, 'Дисперсия не рассчитана', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Дисперсия кригинга')
    
    # 5. Гистограмма исходных значений
    ax5 = axes[1, 1]
    ax5.hist(result['values'], bins=50, alpha=0.7, color='blue', 
            edgecolor='black', label='Исходные точки')
    ax5.axvline(np.mean(result['values']), color='red', linestyle='--',
               label=f'Среднее: {np.mean(result["values"]):.4f}')
    ax5.set_xlabel('Значение')
    ax5.set_ylabel('Частота')
    ax5.set_title('Распределение значений в точках')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 3D визуализация (опционально)
    ax6 = axes[1, 2]
    try:
        from mpl_toolkits.mplot3d import Axes3D
        # Берем небольшой фрагмент для 3D
        size_3d = 100
        if kriged_down.shape[0] > size_3d and kriged_down.shape[1] > size_3d:
            fragment = kriged_down[:size_3d, :size_3d]
            X, Y = np.meshgrid(np.arange(fragment.shape[1]), 
                              np.arange(fragment.shape[0]))
            
            ax6 = fig.add_subplot(236, projection='3d')
            surf = ax6.plot_surface(X, Y, fragment, cmap='viridis', 
                                   alpha=0.8, linewidth=0)
            ax6.set_title('3D поверхность (фрагмент)')
        else:
            ax6.text(0.5, 0.5, 'Недостаточно данных\nдля 3D визуализации', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('3D визуализация')
    except:
        ax6.text(0.5, 0.5, '3D визуализация не доступна', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('3D визуализация')
    
    plt.suptitle('Сравнение исходного растра и кригинга', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print("\nСтатистика сравнения:")
    print(f"  Исходные точки: {len(result['values'])}")
    print(f"  Min значение в точках: {result['values'].min():.4f}")
    print(f"  Max значение в точках: {result['values'].max():.4f}")
    print(f"  Среднее в точках: {result['values'].mean():.4f}")
    
    if kriged_result['kriged_variance'] is not None:
        print(f"\n  Средняя дисперсия кригинга: {np.nanmean(kriged_result['kriged_variance']):.4f}")
        print(f"  Max дисперсия: {np.nanmax(kriged_result['kriged_variance']):.4f}")


# ОСНОВНОЙ СКРИПТ
if __name__ == "__main__":
    # 1. ЗАГРУЗКА ДАННЫХ
    raster_path = r"D:\ml_datasets\Забайкалье\landsat8_PCA\hematite.tif"
    shape_path = r"D:\ml_datasets\Забайкалье\archive\FINAL_ORE_DEPOSITS_AU_AG_CU_MO_porph_ZAB_AMU_Makarov_2025-12-10_EPSG4326_AOI_67.zip"
    
    # Используем ранее написанную функцию для получения индексов
    result = get_point_indices_in_raster(
        raster_path=raster_path,
        shp_path=shape_path
    )
    
    # Загружаем исходный растр для сравнения
    with rs.open(raster_path) as src:
        original_raster = src.read(1)
        raster_shape = original_raster.shape
    
    # 2. ВЫПОЛНЕНИЕ КРИГИНГА
    kriged_result = perform_kriging_interpolation(
        result=result,
        raster_shape=raster_shape,  # сохраняем тот же размер
        output_path="kriging_result.tif",  # сохранить результат
        variogram_model='spherical',  # можно попробовать 'gaussian', 'exponential'
        cell_size=None  # автоматический подбор
    )
    
    # 3. СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    compare_kriging_with_original(
        original_raster=original_raster,
        kriged_result=kriged_result,
        result=result,
        downsample_factor=10
    )
    
    # 4. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ
    print("\n" + "="*60)
    print("ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
    print("="*60)
    
    # Визуализация точек на результате кригинга
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Точки на исходном растре
    ax1 = axes[0]
    with rs.open(raster_path) as src:
        raster_display = src.read(1)[::10, ::10]
    ax1.imshow(raster_display, cmap='viridis')
    ax1.scatter(result['cols']//10, result['rows']//10, 
               c=result['values'], cmap='hot', s=20, alpha=0.7)
    ax1.set_title('Исходные точки на растре')
    
    # Точки на результате кригинга
    ax2 = axes[1]
    kriged_display = kriged_result['kriged_raster'][::10, ::10]
    ax2.imshow(kriged_display, cmap='viridis')
    ax2.scatter(result['cols']//10, result['rows']//10,
               c=result['values'], cmap='hot', s=20, alpha=0.7)
    ax2.set_title('Точки на результате кригинга')
    
    plt.tight_layout()
    plt.show()