import miniaudio
import os


ar = os.listdir("audio3")
for i in ar:
    print(i)
    audio_path = f"audio3/{i}"
    decoded_data = miniaudio.decode_file(audio_path)

    output_path = f"audio4/{i[:-4]}.wav"
    miniaudio.wav_write_file(output_path, decoded_data)

print("OK")