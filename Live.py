import io
import os
import requests
import numpy as np
import librosa
import librosa.display
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

class Live:
    def __init__(self, url, result_label_lab, duration=10, sample_rate=44100, channels=1):
        self.url = url
        self.result_label_lab = result_label_lab
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.n_mels = 128
        self.chunk_size = 2048
        self.overlap_size = int(self.chunk_size * 0.75)
        self.data_queue = []
        self.buffer = bytearray()
        self.fig_lab, self.ax = plt.subplots()
        self.spectr_data = np.zeros((self.n_mels, int(self.sample_rate / (self.chunk_size - self.overlap_size) * self.duration)))
        self.im = self.ax.imshow(self.spectr_data, aspect='auto', origin='lower', cmap='magma')
        plt.colorbar(self.im, ax=self.ax, format='%+2.0f dB')
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Frequency [Hz]')
        self.ax.set_title("Live Mel-frequency spectrogram")
        self.ani = None
        self.model = tf.keras.models.load_model('Model/sea_mammal_classifier.h6')
        self.threshold = 0.8
        self.probka = []
        self.output_path = "Live/Lab"

    def start_live_recognition(self):
        if self.ani is None:
            self.ani = FuncAnimation(self.fig_lab, self.update_plot, interval=100, blit=False, cache_frame_data=False)
            self.stream_audio()
        return self.fig_lab

    def predict_species(self, image_array):
        if image_array is not None:
            predictions = self.model.predict(image_array)
            confidence = np.max(predictions)
            species = np.argmax(predictions)
            species_name = self.get_species_name(species)

            if species_name == "Nie udało się rozpoznać zwierzęcia" or confidence < self.threshold:
                self.result_label_lab.config(text=f"Nie udało się rozpoznać zwierzęcia")
            else:
                self.result_label_lab.config(text=f"Gatunek z hydrofonu: {species_name} (Pewność: {confidence:.2f})")
        else:
            self.result_label_lab.config(text="Błąd: Brak danych do rozpoznania")

    def stream_audio(self):
        response = requests.get(self.url, stream=True)
        for chunk in response.iter_content(chunk_size=1024):
            self.buffer.extend(chunk)
            while len(self.buffer) >= 2:
                audio_data = np.frombuffer(self.buffer[:2], dtype=np.int16)
                self.buffer = self.buffer[2:]
                audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
                self.data_queue.append(audio_data)

    def create_spectrogram(self, data):
        S = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=self.n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB

    def draw_probka(self, spectr, output_path, number):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectr, sr=22050, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.clim(vmin=-70, vmax=0)
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.axis('off')
        filename = f"{number}.png"
        file_path = os.path.join(output_path, filename)
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def update_plot(self, frame):
        if len(self.data_queue) > 0:
            data = np.concatenate(self.data_queue)
            S_DB = self.create_spectrogram(data)
            self.spectr_data = np.roll(self.spectr_data, -1, axis=1)
            self.spectr_data[:, -1] = S_DB.mean(axis=1)
            self.probka.append(self.spectr_data[:, -200:].copy())
            if len(self.probka) % 100 == 0:
                self.draw_probka(self.probka[-1], self.output_path, len(self.probka))
                image_array = self.spectr_data[:, -128:].T
                image_array = np.expand_dims(image_array, axis=-1)
                image_array = np.expand_dims(image_array, axis=0)
                self.predict_species(image_array)
            self.im.set_data(self.spectr_data)
            self.im.set_clim(vmin=-70, vmax=0)
            self.fig_lab.canvas.draw_idle()

    def get_species_name(self, species_id):
        species_dict = {0: "Bieluga", 1: "Delfin", 2: "Delfin Pasiasty", 3: "Delfinowiec", 4: "Humbak", 5: "Kaszalot",
                        6: "Nie udało się rozpoznać zwierzęcia", 7: "Orka", 8: "Wal Grenlandzki"}
        return species_dict.get(species_id, "Nie udało się rozpoznać zwierzęcia")

if __name__ == "__main__":
    url = "https://live.orcasound.net/listen/port-townsend"
    result_label_lab = tk.Label(text="")
    live_recognition = Live(url, result_label_lab)
    live_recognition.start_live_recognition()
