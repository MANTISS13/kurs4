import os
import cv2
import pandas as pd
from fer import FER
import time

# Создаем объект детектора
detector = FER(mtcnn=True)
path_save = 'C:\\Work\\test_set2'

# Подготавливаем пустой DataFrame для хранения результатов
df = pd.DataFrame(columns=['filename', 'predicted_emotion', 'actual_emotion'])
sch = 0

# Словарь для хранения статистики по точности для каждой эмоции
accuracy_stats = {}

# Перебираем файлы в каталоге
for emotion_folder in os.listdir(path_save):
    emotion_path = os.path.join(path_save, emotion_folder)
    if os.path.isdir(emotion_path):
        start_time = time.time()
        sch = 0
        # Перебираем файлы внутри папки с эмоцией
        for filename in os.listdir(emotion_path):
            if filename.endswith('.png'):  # add more formats if needed
                # Загружаем изображение
                image = cv2.imread(os.path.join(emotion_path, filename))
                image = image[:, :, ::-1]
                # Определяем эмоции на изображении
                results = detector.top_emotion(image)

                # Обработка результатов
                if results:
                    sch += 1
                    predicted_emotion = results[0]
                    emotions = {'filename': filename, 'predicted_emotion': predicted_emotion, 'actual_emotion': emotion_folder}
                    df = df.append(emotions, ignore_index=True)

        end_time = time.time()
        op = end_time - start_time
        accuracy = (df[df['actual_emotion'] == emotion_folder]['predicted_emotion'] == emotion_folder).sum() / sch
        accuracy_stats[emotion_folder] = accuracy * 100
        print(f'Processed {sch} frames for {emotion_folder}')
        print(f'Потратили {op} секунд на обработку фото')
        print(f'Accuracy for {emotion_folder} = {accuracy * 100:.2f}%')

# Вывод точности для каждой эмоции после всех папок
for emotion_folder, accuracy in accuracy_stats.items():
    print(f'Accuracy for {emotion_folder} = {accuracy:.2f}%')

# Сохраняем результаты в CSV
df.to_csv(path_save + '\\emotion_results_fer.csv', index=False, sep=';')
