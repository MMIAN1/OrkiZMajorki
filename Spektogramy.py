import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Ścieżki
input_base_dir = "Orka/"
output_base_dir = "Probki/Orka_Spekt"

# Upewnij się, że katalog wyjściowy istnieje
os.makedirs(output_base_dir, exist_ok=True)


def create_spectrogram(audio_path, output_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.axis('off')  # Usuwa osie
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Przejście przez wszystkie pliki w katalogu wejściowym
for root, dirs, files in os.walk(input_base_dir):
    for file in files:
        if file.endswith('.wav'):
            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_base_dir)
            output_folder = os.path.join(output_base_dir, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            output_file_path = os.path.join(output_folder, file.replace('.wav', '.png'))
            create_spectrogram(input_file_path, output_file_path)

print("Konwersja zakończona.")
