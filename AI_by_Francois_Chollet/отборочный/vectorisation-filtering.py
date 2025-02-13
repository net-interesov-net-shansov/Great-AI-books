import pandas as pd
import numpy as np
from tqdm import tqdm

def process_batch(batch_data, pixel_columns, min_threshold=0.05, max_threshold=0.95, similarity_threshold=0.98):
    try:
        # Преобразуем весь батч в numpy array сразу
        pixels = batch_data[pixel_columns].values.astype(float)
        
        # Проверяем шум для всего батча векторизованно
        pixel_density = (pixels != 0).mean(axis=1)
        clean_mask = (pixel_density >= min_threshold) & (pixel_density <= max_threshold)
        
        # Фильтруем шумные изображения
        clean_pixels = pixels[clean_mask]
        clean_data = batch_data[clean_mask].reset_index(drop=True)
        
        if len(clean_pixels) == 0:
            return pd.DataFrame()
            
        # Нормализуем все векторы сразу
        norms = np.linalg.norm(clean_pixels, axis=1)
        valid_mask = norms > 0
        normalized = clean_pixels[valid_mask] / norms[valid_mask, np.newaxis]
        clean_data = clean_data[valid_mask].reset_index(drop=True)
        
        if len(normalized) == 0:
            return pd.DataFrame()
        
        # Считаем матрицу косинусного сходства между всеми парами в батче
        similarities = np.dot(normalized, normalized.T)
        
        # Создаем маску для дубликатов
        # Используем нижнетреугольную матрицу, чтобы всегда оставлять первое изображение из группы дубликатов
        duplicate_mask = ~np.tril(similarities > similarity_threshold, k=-1).any(axis=1)
        
        return clean_data[duplicate_mask].reset_index(drop=True)
        
    except Exception as e:
        print(f"Ошибка в process_batch: {e}")
        return pd.DataFrame()

def filter_and_save_img(original_csv_path, new_csv_file, batch_size=1000):
    original_csv = pd.read_csv(original_csv_path)
    pixel_columns = [col for col in original_csv.columns if col != 'label']
    
    print(f"Начало фильтрации. Всего изображений: {len(original_csv)}")
    
    total_batches = len(original_csv) // batch_size + (1 if len(original_csv) % batch_size != 0 else 0)
    
    # Инициализируем файл с заголовками
    pd.DataFrame(columns=original_csv.columns).to_csv(new_csv_file, index=False)
    
    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(original_csv))
        
        # Получаем и обрабатываем батч целиком
        batch_data = original_csv.iloc[start_idx:end_idx]
        filtered_batch = process_batch(batch_data, pixel_columns)
        
        # Сохраняем результаты построчно
        if not filtered_batch.empty:
            filtered_batch.to_csv(new_csv_file, mode='a', header=False, index=False)
    
    filtered_df = pd.read_csv(new_csv_file)
    print(f"Фильтрация завершена. Сохранено изображений: {len(filtered_df)}")
    
    return filtered_df

def analyse(original_csv_path, filtered_csv_path):
    td = pd.read_csv(original_csv_path)
    ntd = pd.read_csv(filtered_csv_path)
    
    print("\nРезультаты фильтрации:")
    print(f"Исходное кол-во изображений: {len(td)}")
    print(f"Удалено изображений: {len(td) - len(ntd)}")
    print(f"Оставшееся кол-во изображений: {len(ntd)}")
    
    if 'label' in td.columns:
        print("\n---Распределение меток---")
        print("До фильтрации:")
        print(td['label'].value_counts())
        print("\nПосле фильтрации:")
        print(ntd['label'].value_counts())

ntd = filter_and_save_img('fashion-mnist_train.csv', 'new-fashion-mnist_train.csv')
analyse('fashion-mnist_train.csv', 'new-fashion-mnist_train.csv')