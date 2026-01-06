import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from emnist import extract_training_samples, extract_test_samples

# 1. LOAD DATASET (Gunakan split 'balanced' agar distribusi huruf besar/kecil merata)
# Mapping: 0-9 (Angka), 10-35 (Huruf Besar), 36-46 (Huruf Kecil tertentu)
print("Loading EMNIST data...")
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

# 2. PREPROCESSING
# Normalisasi (0-1) dan Reshape ke (28, 28, 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(f"Shape Train: {X_train.shape}, Label Train: {y_train.shape}")
num_classes = len(np.unique(y_train)) # Deteksi jumlah kelas otomatis

# 3. BUILD CNN MODEL
model = Sequential([
    # Conv Block 1
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Conv Block 2
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten & Dense
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Mencegah Overfitting
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAINING
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=128, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# 5. EVALUASI
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Save model untuk dipakai di aplikasi nanti
model.save('ocr_cnn_model.h5')

# 6. MAPPING LABEL (EMNIST Balanced Mapping Ref)
# Perlu kamus mapping manual untuk mengubah angka prediksi (0-46) menjadi karakter (0-9, A-Z, a-z)
# Contoh sederhana mapping label:
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt" 
# Note: EMNIST Balanced merge beberapa huruf kecil dan besar yang mirip (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z)

def predict_image(img_array):
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    return label_map[predicted_label]