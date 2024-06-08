import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Ładowanie modelu
model = load_model('Model/sea_mammal_classifier.h3')

# Klasy modelu (dopasuj do swoich klas)
class_names = ['Delfin', 'Humbak', 'Orka']


# Funkcja do przetwarzania i przewidywania
def predict_image(file_path):
    img = load_img(file_path, target_size=(128, 128), color_mode='grayscale')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction[0])]
    return predicted_class


# Funkcja do ładowania pliku
def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        result_label.config(text=f'Przewidywany gatunek: {prediction}')

        img = Image.open(file_path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img


# Konfiguracja GUI
root = tk.Tk()
root.title("Rozpoznawanie Ssaki Morskie")

frame = tk.Frame(root)
frame.pack(pady=20)

btn = tk.Button(frame, text="Wybierz plik", command=load_file)
btn.pack(pady=10)

result_label = tk.Label(frame, text="Przewidywany gatunek: ", font=("Helvetica", 16))
result_label.pack(pady=10)

panel = tk.Label(frame)
panel.pack(pady=10)

root.mainloop()
