import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from Live import Live
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
        self.configure(bg="lightblue")

        # Load and set background image
        self.bg_image = Image.open('Grafika/background.jpg')
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.bg_label = tk.Label(self, image=self.bg_photo)
        self.bg_label.place(relwidth=1, relheight=1)

        self.main_frame = tk.Frame(self, bg='lightblue')
        self.main_frame.pack(fill='both', expand=True)

        self.title_label = tk.Label(self.main_frame, text="Rozpoznawanie ssaków morskich",
                                    font=("Helvetica", 24, "bold"), bg="#4CAF50", fg="white")
        self.title_label.pack(pady=10)

        self.label = tk.Label(self.main_frame, text="Wybierz obraz spektrogramu, aby rozpoznać gatunek ssaka morskiego",
                              font=("Helvetica", 14), bg="#4CAF50", fg="white")
        self.label.pack(pady=5)

        self.upload_button = tk.Button(self.main_frame, text="Wybierz plik", command=self.load_image,
                                       font=("Helvetica", 12), bg="#4CAF50", fg="white")
        self.upload_button.pack(pady=5)

        self.image_label = tk.Label(self.main_frame, bg='lightblue')
        self.image_label.pack(pady=5)

        self.predict_button = tk.Button(self.main_frame, text="Rozpoznaj gatunek", command=self.on_predict_button_click,
                                        font=("Helvetica", 12), bg="#2196F3", fg="white")
        self.predict_button.pack(pady=5)

        self.live_button = tk.Button(self.main_frame, text="Rozpoznawanie na żywo", command=self.start_live_recognition,
                                     font=("Helvetica", 12), bg="#FF9800", fg="white")
        self.live_button.pack(pady=5)

        self.listen_button = tk.Button(self.main_frame, text="Nasłuchiwanie", command=self.start_listening,
                                       font=("Helvetica", 12), bg="#9C27B0", fg="white")
        self.listen_button.pack(pady=5)

        self.result_frame = tk.Frame(self.main_frame, bg="lightblue")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.result_label = tk.Label(self.result_frame, text="", font=("Helvetica", 12), bg="lightblue")
        self.result_label.pack(pady=5)

        self.result_frame_lab = tk.Frame(self.main_frame, bg="lightblue")
        self.result_frame_lab.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.result_label_lab = tk.Label(self.result_frame_lab, text="", font=("Helvetica", 12), bg="lightblue")
        self.result_label_lab.pack(pady=5)

        try:
            self.model = tf.keras.models.load_model('Model/sea_mammal_classifier.h6')
        except IOError as e:
            print(f"Error loading model: {e}")

        self.threshold = 0.8

        # Initialize matplotlib figures and axes for microphone data
        self.fig_micro, self.ax = plt.subplots()
        self.spectr_data = np.zeros((n_mels, int(fs / (chunk_size - overlap_size) * duration)))
        self.im = self.ax.imshow(self.spectr_data, aspect='auto', origin='lower', extent=[0, duration, 0, fs / 2],
                                 cmap='magma')
        plt.colorbar(self.im, ax=self.ax, format='%+2.0f dB')
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Frequency [Hz]')
        self.ax.set_title("Mel-frequency spectrogram")

        self.canvas_micro = FigureCanvasTkAgg(self.fig_micro, master=self.result_frame)
        self.canvas_micro.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

        # Initialize the live recognition with result_label_lab
        self.live_recognition = Live("https://live.orcasound.net/listen/port-townsend", self.result_label_lab)
        self.fig_lab = self.live_recognition.fig_lab
        self.canvas_lab = FigureCanvasTkAgg(self.fig_lab, master=self.result_frame_lab)
        self.canvas_lab.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

        self.output_path = "Live/Micro"
        self.probka = []
        self.image_array = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).convert('L')
            self.image = self.image.resize((128, 128), Image.LANCZOS)  # Resize to exact 128x128 dimensions
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.image_tk)
            self.result_label.config(text="")

            # Prepare the image array for prediction
            image = np.array(self.image)
            image = image / 255.0  # Normalize the image
            image = np.expand_dims(image, axis=(0, -1))  # Expand dimensions to fit the model input shape
            self.image_array = image

    def on_predict_button_click(self):
        self.predict_species(self.image_array)

    def predict_species(self, image_array):
        if image_array is not None:
            predictions = self.model.predict(image_array)
            confidence = np.max(predictions)
            species = np.argmax(predictions)
            species_name = self.get_species_name(species)

            if species_name == "Nie udało się rozpoznać zwierzęcia" or confidence < self.threshold:
                self.result_label.config(text=f"Nie udało się rozpoznać zwierzęcia")
            else:
                self.result_label.config(text=f"Rozpoznany gatunek: {species_name} (Pewność: {confidence:.2f})")
        else:
            messagebox.showerror("Błąd", "Brak danych do rozpoznania")

    def start_live_recognition(self):
        self.fig_lab = self.live_recognition.start_live_recognition()
        self.canvas_lab.draw()

    def start_listening(self):
        self.start_audio_stream()
        self.ani = FuncAnimation(self.fig_micro, self.update_plot, interval=100, blit=False, cache_frame_data=False)
        self.canvas_micro.draw()

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
        noise_threshold = -70
        S_DB[S_DB < noise_threshold] = noise_threshold
        return S_DB

    def draw_probka(self, spectr, output_path, number):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectr, sr=fs, x_axis='time', y_axis='mel', cmap='magma')
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

    def update_plot(self, frame, sr=fs):
        while not data_queue.empty():
            data = data_queue.get()
            S_DB = self.create_spectrogram(data, sr=sr)
            self.spectr_data = np.roll(self.spectr_data, -1, axis=1)
            self.spectr_data[:, -1] = S_DB.mean(axis=1)
            self.probka.append(self.spectr_data[:, -200:].copy())
            print(f"Mamy {len(self.probka)} próbek")
            if len(self.probka) % 100 == 0:
                self.draw_probka(self.probka[-1], self.output_path, len(self.probka))
                image_array = self.spectr_data[:, -128:].T
                image_array = np.expand_dims(image_array, axis=-1)
                image_array = np.expand_dims(image_array, axis=0)
                self.predict_species(image_array)
            self.im.set_data(self.spectr_data)
            self.im.set_clim(vmin=-70, vmax=0)
            self.canvas_micro.draw()

    def get_species_name(self, species_id):
        species_dict = {0: "Bieluga", 1: "Delfin", 2: "Delfin Pasiasty", 3: "Delfinowiec", 4: "Humbak", 5: "Kaszalot",
                        6: "Nie udało się rozpoznać zwierzęcia", 7: "Orka", 8: "Wal Grenlandzki"}
        return species_dict.get(species_id, "Nie udało się rozpoznać zwierzęcia")


if __name__ == "__main__":
    app = MarineMammalRecognizer()
    app.mainloop()
