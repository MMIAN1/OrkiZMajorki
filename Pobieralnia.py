import re
import requests
import os

file_path = "Probki_Wav/delfinowiec.txt"
output_directory = "Probki_Wav/Delfinowiec/"

# Utwórz katalog, jeśli nie istnieje
os.makedirs(output_directory, exist_ok=True)

with open(file_path, 'r') as file:
    content = file.read()

# Wyszukaj wszystkie adresy URL za pomocą wyrażenia regularnego
matches = re.findall(r'href="(\/science\/B\/whalesounds\/WhaleSounds\/(\d+[A-Z]*)\.wav)"', content)

# Pobierz pliki .wav
for match in matches:
    url, file_name = match
    full_url = f"https://cis.whoi.edu{url}"
    destination_file = os.path.join(output_directory, f"Delfinowiec{file_name}.wav")

    response = requests.get(full_url)

    if response.status_code == 200:
        with open(destination_file, 'wb') as f:
            f.write(response.content)
        print(f"Pobrano plik {file_name}.wav i zapisano jako {destination_file}")
    else:
        print(f"Nie udało się pobrać pliku {file_name}.wav. Status code: {response.status_code}")