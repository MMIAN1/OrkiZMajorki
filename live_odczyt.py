import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import queue
import librosa
import os
from Rozpoznawanie import AI

fs = 22050
chunk_size = 2048
overlap_size = int(chunk_size * 0.75)
duration = 20
data_queue = queue.Queue()
n_mels = 128
output_path = "Probki/Live"
probka = []

fig, ax = plt.subplots()
spectr_data = np.zeros((n_mels, int(fs / (chunk_size - overlap_size) * duration)))
im = ax.imshow(spectr_data, aspect='auto', origin='lower', extent=[0, duration, 0, fs/2])
plt.colorbar(im, ax=ax)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [Hz]')
ax.set_title("Mel-frequency spectrogram")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    data_queue.put(indata[:, 0])


def create_spectrogram(data, sr=22050):
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def draw_probka(spectr, output_path, number):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectr, sr=fs, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    filename = f"{number}.png"
    file_path = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def sluchanie():
    sluchacz = AI()
    odczyty = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            odczyty.append(file)

    file_path = os.path.join(output_path, odczyty[-1])
    predicted_class, max_pred = sluchacz.predict_image(file_path)
    print(f"Rozpoznany gatunek:{predicted_class}, pewność: {max_pred}")


def update_plot(frame, sr=fs):
    global spectr_data, im, probka
    while not data_queue.empty():
        data = data_queue.get()
        S_DB = create_spectrogram(data, sr=sr)
        spectr_data = np.roll(spectr_data, -1, axis=1)  # przsuwa wykres w lewo
        spectr_data[:, -1] = S_DB.mean(axis=1)
        probka.append(spectr_data[:, -200:].copy())
        if len(probka) % 100 == 0:
            draw_probka(probka[-1], output_path, len(probka))
            sluchanie()
        im.set_data(spectr_data)
        im.set_clim(vmin=np.min(spectr_data), vmax=np.percentile(spectr_data, 95))
        plt.draw()


stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=chunk_size)
stream.start()

ani = FuncAnimation(fig, update_plot, interval=10, blit=False)

plt.show()
