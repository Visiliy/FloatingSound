import sqlite3
import yt_dlp
from demucs.separate import main as demucs_separate
import time
from pydub import AudioSegment
import os
import torch
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import re

ffmpeg_dir = "C:/Users/natal/Downloads/ffmpeg-7.1.1-full_build/ffmpeg-7.1.1-full_build/bin"  # Например: "C:\\Program Files\\ffmpeg\\bin"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
AudioSegment.ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_dir, "ffprobe.exe")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
s = 0


def vocal2notes(path0):
    try:

        demucs_separate([path0, "--two-stems", "vocals", '-o', 'pesni/', '-d', device])

        # demucs_separate([
        # input_audio,
        # '-o', output_folder,
        # '-n',
        # '--two-stems=vocals',
        # '--float32',
        # '-d',
        # device
        # ])
        hhh = path0.split('/')[1].split('.')[0]
        path = f'pesni/htdemucs/{hhh}/vocals.wav'
        model_output, midi_data, note_events = predict(path, onset_threshold=0.85, frame_threshold=0.3,
                                                       maximum_frequency=1000)

        global s
        s += 1
        notes = midi_data.instruments[0].notes
        notes_new = []
        last_note = 'zjdajh'
        for i in notes:
            if last_note != i:
                notes_new.append(i.pitch)
                # print(pretty_midi.note_number_to_name(i.pitch), end=' ')
            last_note = i

        differences = [notes_new[i + 1] - notes_new[i]
                       for i in range(len(notes_new) - 1)]
        return differences
    except Exception as e:
        return e


def download_audio(path, song_name, output_dir="pesni/"):
    """Скачивает аудио с YouTube по названию песни."""
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}{song_name}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f"ytsearch1:{path}"])  # Ищет первое совпадение на YouTube
            return f"{output_dir}{song_name}.mp3"
        except Exception as e:
            print('не удалось загрузить ', song_name)
            print(f"Ошибка при скачивании {song_name}: {e}")
            return None


def conve():
    ffmpeg_dir = "C:/Users/natal/Downloads/ffmpeg-7.1.1-full_build/ffmpeg-7.1.1-full_build/bin"  # Например: "C:\\Program Files\\ffmpeg\\bin"
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

    AudioSegment.ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(ffmpeg_dir, "ffprobe.exe")

    audio = AudioSegment.from_file(f"pesni/audio.webm", format="webm")
    audio.export("audio.wav", format="wav")


# Пример использования:

conn = sqlite3.connect("music.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM music;")
rows = cursor.fetchall()
