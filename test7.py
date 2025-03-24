import librosa
import sqlite3
import json
from music21 import pitch, stream, note, tempo
from time import time

# Загрузка аудиофайла
def f585(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Извлечение высоты тона
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    # Получение доминирующих частот
    dominant_frequencies = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_freq = pitches[index, t]
        if pitch_freq > 0:
            dominant_frequencies.append(pitch_freq)

    # Преобразование частот в ноты
    notes = []
    for freq in dominant_frequencies:
        p = pitch.Pitch()
        p.frequency = freq
        notes.append(p.nameWithOctave)

    # Создание партитуры
    score = stream.Score()
    part = stream.Part()

    # Добавление нот в партитуру
    for n in notes:
        n = note.Note(n)
        part.append(n)

    # Добавление темпа
    part.insert(0, tempo.MetronomeMark(number=120))
    print(len(notes))
    return notes

new_n = f585(f"media_files/recording_20250323_082759.wav")

connection = sqlite3.connect('my_database.db')
cursor = connection.cursor()

cursor.execute('SELECT * FROM Users')
music = cursor.fetchall()
name = ""
main_coeff = 0
for s in music:
    t1 = time()
    ar2 = json.loads(s[-1])
    print(f'json loaded with time {time() - t1}')
    t2 = time()
    ar1 = new_n
    max_coeff = 0
    for i in range((len(ar2) - len(ar1)) // 2):
        coeff = 0
        for j in range(0, len(ar1)):
            if ar1[j] == ar2[i + j]:
                coeff += 1
        max_coeff = max(coeff, max_coeff)
    if max_coeff >= main_coeff:
        main_coeff = max_coeff
        name = s[1]
    print(f'arr processed with time {time() - t2}')
print(name)
connection.close()

