import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model

# 1. LOAD MODEL
# Pastikan file .h5 ada di folder yang sama dengan script ini
model_path = 'cnn_ocr_emnist_csv.h5'
if not os.path.exists(model_path):
    print(f"Error: Model '{model_path}' tidak ditemukan!")
    exit()

model = load_model(model_path)
print("Model berhasil dimuat!")

# 2. MAPPING LABEL (EMNIST Balanced)
label_map = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z',
    36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'
}

def predict_custom_image(image_path):
    # a. Baca gambar asli
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print("Gambar tidak ditemukan!")
        return

    # b. Ubah ke Grayscale
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # c. Resize ke 28x28 (Sesuai input model)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # d. INVERT & THRESHOLD (PENTING!)
    # Mengubah background putih jadi hitam, dan tulisan hitam jadi putih terang
    # Kita gunakan OTSU threshold agar pemisahan warna lebih tegas
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # e. Normalisasi
    final_img = thresh.astype('float32') / 255.0
    
    # f. Reshape untuk model (1, 28, 28, 1)
    input_data = final_img.reshape(1, 28, 28, 1)

    # g. Prediksi
    prediction = model.predict(input_data)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    label = label_map.get(result_index, "?")

    # Tampilkan Hasil
    print(f"\n--- HASIL PREDIKSI ---")
    print(f"Karakter : {label}")
    print(f"Keyakinan: {confidence:.2f}%")
    
    plt.imshow(final_img, cmap='gray')
    plt.title(f"Prediksi: {label} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()

# --- EKSEKUSI ---
# Ganti dengan nama file gambar Anda
nama_file = 'Screenshot 2026-01-02 145536.png' 
predict_custom_image(nama_file)