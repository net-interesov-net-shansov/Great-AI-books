import pandas as pd
import numpy as np
from tqdm import tqdm

def find_noisy_images(pixels, min_threshold=0.05, max_threshold=0.95):
    try:
        pixels_array = np.array(pixels).astype(float)
        pixel_density = (pixels_array != 0).mean()
        return pixel_density < min_threshold or pixel_density > max_threshold
    except Exception:
        return True

def find_duplicates_in_batch(image_data, batch_pixels, similarity_threshold=0.98):
    if len(batch_pixels) == 0:
        return False
    
    try:
        # Сравниваем с изображениями только в текущем батче
        image_data_array = np.array(image_data).astype(float)
        batch_pixels_array = np.array(batch_pixels).astype(float)
        
        # Вычисляем нормы
        norm_image = np.linalg.norm(image_data_array)
        norms_batch = np.linalg.norm(batch_pixels_array, axis=1)
        
        # Вычисляем косинусное сходство
        dot_products = np.dot(batch_pixels_array, image_data_array)
        similarities = dot_products / (norms_batch * norm_image)
        
        return np.any(similarities > similarity_threshold)
    except Exception as e:
        print(f"Ошибка в find_duplicates_in_batch: {e}")
        return False

def filter_and_save_img(original_csv_path, new_csv_file, batch_size=1000):
    original_csv = pd.read_csv(original_csv_path)
    pixel_columns = [col for col in original_csv.columns if col != 'label']
    
    filtered_data_list = []
    print(f"Начало фильтрации. Всего изображений: {len(original_csv)}")
    
    total_batches = len(original_csv) // batch_size + (1 if len(original_csv) % batch_size != 0 else 0)
    
    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(original_csv))
        
        batch_data = original_csv.iloc[start_idx:end_idx]
        batch_filtered = []
        batch_pixels = []
        
        for _, row in batch_data.iterrows():
            pixels = row[pixel_columns].values
            
            # Проверка на шум
            if find_noisy_images(pixels):
                continue
            
            # Проверка на дубликаты только внутри текущего батча
            if find_duplicates_in_batch(pixels, batch_pixels):
                continue
            
            batch_filtered.append(row)
            batch_pixels.append(pixels)
        
        # Сохраняем отфильтрованные данные из батча
        if batch_filtered:
            filtered_data_list.extend(batch_filtered)
        
        # Периодическая запись в файл
        if len(filtered_data_list) >= batch_size:
            temp_df = pd.DataFrame(filtered_data_list)
            if not pd.io.common.file_exists(new_csv_file):
                temp_df.to_csv(new_csv_file, index=False)
            else:
                temp_df.to_csv(new_csv_file, mode='a', header=False, index=False)
            filtered_data_list = []
    
    # Записываем оставшиеся данные
    if filtered_data_list:
        temp_df = pd.DataFrame(filtered_data_list)
        if not pd.io.common.file_exists(new_csv_file):
            temp_df.to_csv(new_csv_file, index=False)
        else:
            temp_df.to_csv(new_csv_file, mode='a', header=False, index=False)
    
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