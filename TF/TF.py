import numpy as np
import cv2
import tensorflow as tf
import time
import os
import pandas as pd

path_save = 'C:\\Work\\test_set'
settings = {
    'scaleFactor': 1.3, 
    'minNeighbors': 5, 
    'minSize': (60, 60)
}
df = pd.DataFrame({'Эмоция': [], 'Папка': []})

face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml') #классификатор модели 
labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad'] # набор эмоций
model = tf.keras.models.load_model('network-5Labels.h5') #модель

# Перебираем файлы в каталоге
for emotion_folder in os.listdir(path_save):
    emotion_path = os.path.join(path_save, emotion_folder)
    if os.path.isdir(emotion_path):
        sch = 0
        # Перебираем файлы внутри папки с эмоцией
        for filename in os.listdir(emotion_path):
            if filename.endswith('.png'):  # add more formats if needed
                image = cv2.imread(os.path.join(emotion_path, filename))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected = face_detection.detectMultiScale(gray, **settings)
                for x, y, w, h in detected:
                    cv2.rectangle(gray, (x, y), (x+w, y+h), (245, 135, 66), 2)
                    cv2.rectangle(gray, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
                    face = gray[y+5:y+h-5, x+20:x+w-20]
                    face = cv2.resize(face, (48,48)) 
                    face = face/255.0

                    predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
                    state = labels[predictions]

                    # Обработка результатов
                    if state:
                        sch += 1
                        df_new_row = pd.DataFrame({'Эмоция': [state], 'Папка': [emotion_folder]})
                        df = pd.concat([df, df_new_row], ignore_index=True)

        print(f'Processed {sch} frames for {emotion_folder}')

# Вывод точности для каждой эмоции после всех папок
for emotion_folder in df['Папка'].unique():
    sch = (df['Папка'] == emotion_folder).sum()
    if sch > 0:
        accuracy = (df[df['Папка'] == emotion_folder]["Эмоция"] == emotion_folder).sum() / sch
        print(f'Accuracy for {emotion_folder} = {accuracy * 100:.2f}%')

# Сохраняем результаты в CSV
df.to_csv(path_save+'\\emotion_results_tf.csv', index=False, sep=';')
