# DINO Labeler

Скрипт для детекции объектов с помощью DINO. Работает в Google Colab.

## Установка в Colab

1. Создайте токен на GitHub:
   - Зайдите в Settings -> Developer settings -> Personal access tokens
   - Выберите "Tokens (classic)"
   - Generate new token
   - Поставьте галочку на `repo`
   - Скопируйте токен

2. В Colab выполните:
```python
# Подставьте свой токен
token = "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"

# Устанавливаем библиотеку из приватного репозитория
!pip install git+https://{token}@github.com/smicurin474/s.git

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