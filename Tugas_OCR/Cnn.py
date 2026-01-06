import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical

# ==========================================
# 1. FUNGSI LOAD & PREPROCESS DATA CSV
# ==========================================
def load_emnist_csv(csv_path):
    print(f"Membaca file {csv_path} (ini mungkin memakan waktu)...")
    df = pd.read_csv(csv_path, header=None)
    
    # Kolom 0 adalah Label, Sisanya adalah Pixel
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Normalisasi (0-255 -> 0-1)
    X = X.astype('float32') / 255.0
    
    # Reshape dari 784 (flat) menjadi (28, 28)
    # Catatan: Data CSV EMNIST seringkali terputar. Kita perlu memutarnya kembali.
    X_reshaped = []
    for i in range(len(X)):
        # Reshape ke 28x28
        img = X[i].reshape(28, 28)
        # Transpose (Putar) dan Flip agar orientasi benar
        img = np.fliplr(img)
        img = np.rot90(img)
        X_reshaped.append(img)
    
    X_reshaped = np.array(X_reshaped)
    
    # Tambah dimensi channel menjadi (JumlahData, 28, 28, 1) untuk CNN
    X_reshaped = X_reshaped.reshape(-1, 28, 28, 1)
    
    return X_reshaped, y

# ==========================================
# 2. EKSEKUSI LOAD DATA
# ==========================================
# Ganti nama file sesuai yang Anda download dari Kaggle
train_path = 'emnist-balanced-train.csv' 
test_path = 'emnist-balanced-test.csv'

try:
    X_train, y_train = load_emnist_csv(train_path)
    X_test, y_test = load_emnist_csv(test_path)
    print("Data berhasil dimuat!")
    print(f"Shape Train: {X_train.shape}")
    print(f"Shape Test: {X_test.shape}")
except FileNotFoundError:
    print("ERROR: File CSV tidak ditemukan. Pastikan nama file dan path sudah benar.")
    exit()

# Mapping Label (Untuk EMNIST Balanced - 47 Kelas)
# 0-9: Angka
# 10-35: Huruf Besar (A-Z)
# 36-46: Huruf Kecil (a, b, d, e, f, g, h, n, q, r, t) -> Sisanya digabung ke huruf besar
label_map = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z',
    36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'
}

# Cek Visualisasi Data Random
plt.figure(figsize=(10, 2))
for i in range(5):
    idx = np.random.randint(0, len(X_train))
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    label_index = y_train[idx]
    plt.title(f"Label: {label_map.get(label_index, '?')}")
    plt.axis('off')
plt.show()

# ==========================================
# 3. MEMBANGUN MODEL CNN
# ==========================================
num_classes = 47 # Untuk EMNIST Balanced

model = Sequential([
    # Layer Konvolusi 1
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Layer Konvolusi 2
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten & Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Mencegah overfitting
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 4. TRAINING MODEL
# ==========================================
print("Mulai Training CNN...")
history = model.fit(X_train, y_train, 
                    epochs=10, # Bisa ditambah jika akurasi kurang
                    batch_size=128, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# ==========================================
# 5. EVALUASI
# ==========================================
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {score[0]}')
print(f'Test Accuracy: {score[1]*100:.2f}%')

# Simpan Model
model.save('cnn_ocr_emnist_csv.h5')
print("Model berhasil disimpan sebagai 'cnn_ocr_emnist_csv.h5'")