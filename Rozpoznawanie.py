import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Ładowanie modelu
model = load_model('Model/sea_mammal_classifier.h6')


# Klasy modelu (dopasuj do swoich klas)
class_names = ['Bieluga', 'Delfin', 'Delfinek Pasiasty', 'Delfinowiec', 'Humbak', 'Kaszalot', 'Nie wykryto', 'Orka', 'Wal Grenlandzki']

# Próg pewności
threshold = 0.5

dir_path = "Tests/Orka_test"


# Funkcja do przetwarzania i przewidywania
def predict_image(file_path):
    img = load_img(file_path, target_size=(128, 128), color_mode='grayscale')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    max_pred = np.max(prediction[0])
    if max_pred < threshold:
        predicted_class = class_names[-1]
    else:
        predicted_class = class_names[np.argmax(prediction[0])]
    return predicted_class


def tester(dir_path):
    predictions = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                prediction = predict_image(file_path)
                print(f"Predicted sea mammal: {prediction}")
                predictions.append(prediction)


def accuracy(dir_path):
    acc_preds = 0
    preds = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                filename = os.path.basename(file).lower()
                if "delfin" in filename:
                    acc_pred = "delfin"
                if "humbak" in filename:
                    acc_pred = "humbak"
                if "orka" in filename:
                    acc_pred = "orka"
                prediction = predict_image(file_path)
                if prediction.lower() == acc_pred:
                    acc_preds += 1
                preds += 1

    print(f"Predicted {acc_preds} out of {preds}")


accuracy(dir_path)

# Funkcja do ładowania pliku
# def load_file():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         prediction = predict_image(file_path)
#         result_label.config(text=f'Przewidywany gatunek: {prediction}')
#
#         img = Image.open(file_path)
#         img = img.resize((300, 300), Image.ANTIALIAS)
#         img = ImageTk.PhotoImage(img)
#         panel.config(image=img)
#         panel.image = img


# Konfiguracja GUI
# root = tk.Tk()
# root.title("Rozpoznawanie Ssaki Morskie")
#
# frame = tk.Frame(root)
# frame.pack(pady=20)
#
# btn = tk.Button(frame, text="Wybierz plik", command=load_file)
# btn.pack(pady=10)
#
# result_label = tk.Label(frame, text="Przewidywany gatunek: ", font=("Helvetica", 16))
# result_label.pack(pady=10)
#
# panel = tk.Label(frame)
# panel.pack(pady=10)
#
# root.mainloop()
