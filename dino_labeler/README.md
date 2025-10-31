# DINO Labeler

Скрипт для детекции объектов с помощью DINO. Работает в Google Colab.

## Установка в Colab

```python
# Устанавливаем библиотеку из GitHub
!pip install git+https://github.com/smicurin474/s.git

# Если нужны зависимости
!pip install torch transformers pillow tqdm pandas
```

## Как использовать

```python
from google.colab import drive
drive.mount('/content/drive')

from dino_labeler.labeler import run_detector

# Путь к папке в Google Drive
base_dir = '/content/drive/MyDrive/ML отдел/LM ОКК/Удлинитель'

# Папки которые надо пропустить
excluded_dirs = ['Контр. выборка', 'Виды удлинителей.pdf']

# Что искать на фотографиях
prompt = 'удлинитель'

# Запускаем детектор
df = run_detector(base_dir, excluded_dirs, prompt)

# Сохраняем результаты
df.to_csv('detection_results.csv', index=False, encoding='utf-8')
```