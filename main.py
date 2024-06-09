import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from Live import start_live_recognition
import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sounddevice as sd
import queue
import librosa
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

fs = 22050
chunk_size = 2048
overlap_size = int(chunk_size * 0.75)
duration = 20
data_queue = queue.Queue()
n_mels = 128


class MarineMammalRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Rozpoznawanie ssaków morskich")
        self.geometry("1280x720")

        self.label = tk.Label(self, text="Wybierz obraz spektrogramu, aby rozpoznać gatunek ssaka morskiego")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self, text="Wybierz plik", command=self.load_image)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.predict_button = tk.Button(self, text="Rozpoznaj gatunek", command=self.predict_species)
        self.predict_button.pack(pady=10)

        self.live_button = tk.Button(self, text="Rozpoznawanie na żywo", command=self.start_live_recognition)
        self.live_button.pack(pady=10)

        self.listen_button = tk.Button(self, text="Nasłuchiwanie", command=self.start_listening)
        self.listen_button.pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=20)

        try:
            self.model = tf.keras.models.load_model('Model/sea_mammal_classifier.h6')
        except IOError as e:
            print(f"Error loading model: {e}")

        self.threshold = 0.8

        # Initialize matplotlib figure and axes
        self.fig, self.ax = plt.subplots()
        self.spectr_data = np.zeros((n_mels, int(fs / (chunk_size - overlap_size) * duration)))
        self.im = self.ax.imshow(self.spectr_data, aspect='auto', origin='lower', extent=[0, duration, 0, fs/2])
        plt.colorbar(self.im, ax=self.ax)
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Frequency [Hz]')
        self.ax.set_title("Mel-frequency spectrogram")

        # Create a canvas for the matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).convert('L')  # Konwertuj obraz do skali szarości
            self.image.thumbnail((300, 300))
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.image_tk)
            self.result_label.config(text="")

    def predict_species(self):
        if hasattr(self, 'image'):
            # Przygotuj obraz do przewidywania
            image_array = np.array(self.image.resize((128, 128))) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)
            image_array = np.expand_dims(image_array, axis=0)

            # Przewidywanie
            predictions = self.model.predict(image_array)
            confidence = np.max(predictions)
            species = np.argmax(predictions)
            species_name = self.get_species_name(species)

            if self.get_species_name(species) == "Nie udało się rozpoznać zwierzęcia":
                self.result_label.config(text=f"Nie udało się rozpoznać zwierzęcia")
            elif confidence < self.threshold:
                self.result_label.config(text=f"Nie udało się rozpoznać zwierzęcia")
            else:
                self.result_label.config(text=f"Rozpoznany gatunek: {species_name} (Pewność: {confidence:.2f})")
        else:
            messagebox.showerror("Błąd", "Najpierw wybierz obraz spektrogramu")

    def start_live_recognition(self):
        start_live_recognition()

    def start_listening(self):
        self.start_audio_stream()
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False, cache_frame_data=False)
        self.canvas.draw()

    def start_audio_stream(self):
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=fs, blocksize=chunk_size)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        data_queue.put(indata[:, 0])

    def create_spectrogram(self, data, sr=22050):
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB

    def draw_probka(self, spectr, output_path, number):
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

    def update_plot(self, frame, sr=fs):
        self.output_path = "Live"
        self.probka = []
        while not data_queue.empty():
            data = data_queue.get()
            S_DB = self.create_spectrogram(data, sr=sr)
            self.spectr_data = np.roll(self.spectr_data, -1, axis=1)  # przesuwa wykres w lewo
            self.spectr_data[:, -1] = S_DB.mean(axis=1)
            self.probka.append(self.spectr_data[:, -100:].copy())
            if len(self.probka) % 10 == 0:
                self.draw_probka(self.probka[-1], self.output_path, len(self.probka))
            self.im.set_data(self.spectr_data)
            self.im.set_clim(vmin=np.min(self.spectr_data), vmax=np.percentile(self.spectr_data, 95))
            self.canvas.draw()

    def get_species_name(self, species_id):
        species_dict = {0: "Bieluga", 1: "Delfin", 2: "Delfin Pasiasty", 3: "Delfinowiec", 4: "Humbak", 5: "Kaszalot", 6: "Nie udało się rozpoznać zwierzęcia", 7: "Orka", 8: "Wal Grenlandzki"}
        return species_dict.get(species_id, "Nie udało się rozpoznać zwierzęcia")


if __name__ == "__main__":
    app = MarineMammalRecognizer()
    app.mainloop()
