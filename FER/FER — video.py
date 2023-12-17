import os
import cv2
import pandas as pd
from fer import FER
import time

# Создаем объект детектора
detector = FER(mtcnn=True)
path_save = 'C:\\Work'

# Подготавливаем пустой DataFrame для хранения результатов
df = pd.DataFrame(columns=['filename', 'predicted_emotion'])
sch = 0

# Словарь для хранения статистики по точности для каждой эмоции
accuracy_stats = {}
start_time = time.time()
emotion_path = 'C:\\Work\\VAK_IM'
# Перебираем файлы внутри папки с эмоцией
for filename in os.listdir(emotion_path):
    if filename.endswith('jpg'):  # add more formats if needed
        # Загружаем изображение
        image = cv2.imread(os.path.join(emotion_path, filename))
        image = image[:, :, ::-1]
        # Определяем эмоции на изображении
        results = detector.top_emotion(image)
        print(results)
        # Обработка результатов
        if results[1] is not None:
            sch += 1
            predicted_emotion = results[0]
            emotions = {'filename': filename, 'predicted_emotion': predicted_emotion}
            df = df.append(emotions, ignore_index=True)

end_time = time.time()
op = end_time - start_time
print(f'Processed {sch} frames for {emotion_path}')
print(f'Потратили {op} секунд на обработку фото')

# Сохраняем результаты в CSV
df.to_csv(path_save + '\\emotion_results_fer_vid.csv', index=False, sep=';')
