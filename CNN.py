import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parametry
input_shape = (128, 128, 1)  # Zakładając, że spektrogramy są skalowane do 128x128 pikseli

# Ścieżka do katalogu ze spektrogramami
data_dir = 'Probki/'

# Parametry wczytywania
img_width, img_height = 128, 128  # Wymiary spektrogramu
batch_size = 32

# Konfiguracja ImageDataGenerator do wczytywania i augmentacji danych
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizacja pikseli
    validation_split=0.2  # 20% danych jako zbiór walidacyjny
)

# Generator danych treningowych
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generator danych walidacyjnych
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Sprawdzenie liczby klas
num_classes = len(train_generator.class_indices)
print(f"Classes: {train_generator.class_indices}")
print(f"Number of classes: {num_classes}")

# Tworzenie modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=1
)

# Ocena modelu
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy}')
print(f"Classes: {train_generator.class_indices}")


model.save('Model/sea_mammal_classifier.h4o')
