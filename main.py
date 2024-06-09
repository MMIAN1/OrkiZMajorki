import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from Live import start_live_recognition  # Importowanie funkcji z Live.py


class MarineMammalRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Rozpoznawanie ssaków morskich")
        self.geometry("600x500")

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

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=20)

        self.model = tf.keras.models.load_model('Model/sea_mammal_classifier.h4')  # Załaduj model tutaj

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
            species = np.argmax(predictions)
            species_name = self.get_species_name(species)
            self.result_label.config(text=f"Rozpoznany gatunek: {species_name}")
        else:
            messagebox.showerror("Błąd", "Najpierw wybierz obraz spektrogramu")

    def start_live_recognition(self):
        start_live_recognition()  # Wywołaj funkcję rozpoznawania na żywo

    def get_species_name(self, species_id):
        species_dict = {0: "Delfin", 1: "Humbak", 2: "Nie rozpoznano", 3: "Orka"}
        return species_dict.get(species_id, "Nieznany gatunek")


if __name__ == "__main__":
    app = MarineMammalRecognizer()
    app.mainloop()
