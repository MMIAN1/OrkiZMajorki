import os
import wave
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def record_audio(output_filename, duration=10, sample_rate=44100, channels=1):
    print("Nagrywanie rozpoczęte...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()  # Czeka, aż nagranie się zakończy
    print("Nagrywanie zakończone.")

    # Zapisz nagranie jako plik WAV
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())

    print(f"Nagranie zapisane jako {output_filename}")


def create_spectrogram(audio_path, output_dir, sr=22500):
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Poprawne wywołanie z keyword arguments
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.axis('off')  # Usuwa osie
    # Ustal ścieżkę do zapisu
    base_name = os.path.basename(audio_path)
    file_name, _ = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, file_name + '.png')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Spektrogram zapisany jako {output_path}")


def start_live_recognition():
    # Główna część skryptu
    output_filename = f"Live1.wav"
    spectrogram_dir = "Probki_Testowe/NaZywo/"

    # Upewnij się, że katalog do zapisu spektrogramów istnieje
    os.makedirs(spectrogram_dir, exist_ok=True)

    # Podaj URL, który chcesz nagrywać (użyj go w przeglądarce)
    url = "https://live.orcasound.net/listen/port-townsend"
    print(f"Otwórz stronę: {url} w przeglądarce i rozpocznij odtwarzanie dźwięku.")

    # Nagrywaj dźwięk z systemu
    record_audio(output_filename)

    # Twórz spektrogram z nagranego pliku
    create_spectrogram(output_filename, spectrogram_dir)
    os.remove(output_filename)
