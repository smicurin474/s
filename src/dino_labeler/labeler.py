import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
from tqdm import tqdm
import hashlib
import pandas as pd

def run_detector(base_dir, excluded_dirs, prompt):
    """
    Запуск детектора DINO.
    
    Args:
        base_dir: путь к корневой папке с данными
        excluded_dirs: список папок которые надо пропустить
        prompt: промпт для DINO (что искать)
    
    Returns:
        pandas.DataFrame с уникальными хешами
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Загрузка модели Grounding DINO...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def calc_hash256(filepath):
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_image(image_path, text_prompt="tube.", box_threshold=0.25, text_threshold=0.25):
        image_pil = Image.open(image_path).convert("RGB")
        image_width, image_height = image_pil.size

        inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold,
            target_sizes=[(image_height, image_width)]
        )

        if len(results[0]['boxes']) > 0:
            max_score_idx = torch.argmax(results[0]['scores'])
            box = results[0]['boxes'][max_score_idx].tolist()
            # Конвертация в YOLO формат
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            return True, [x_center, y_center, width, height]
        
        return False, []

# Создаем DataFrame для хранения результатов
results_data = []

for dir_name in os.listdir(base_dir):
    if dir_name not in excluded_dirs and os.path.isdir(os.path.join(base_dir, dir_name)):
        input_dir = os.path.join(base_dir, dir_name)
        
        image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        for image_file in tqdm(image_files, desc=f"Обработка {dir_name}"):
            image_path = os.path.join(input_dir, image_file)
            
            # Вычисляем хэш исходного изображения
            image_hash = calc_hash256(image_path)
            
            # Обрабатываем изображение
            detected, yolo_bbox = process_image(image_path)
            
            # Сохраняем результаты
            results_data.append({
                'image_hash': image_hash,
                'tube_detected': detected,
                'yolo_bbox': yolo_bbox if detected else [],
                'original_filename': image_file,
                'source_directory': dir_name
            })
        
        print(f"Обработка {dir_name} завершена.")

# Создаем DataFrame и убираем дубликаты по хешу
df = pd.DataFrame(results_data)
unique_hash_df = df.drop_duplicates(subset=['image_hash'])

print(f"\nВсего обработано изображений: {len(results_data)}")
print(f"Найдено объектов: {df['tube_detected'].sum()}")
print(f"Количество уникальных хешей: {len(unique_hash_df)}")

return unique_hash_df