import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from keras.models import load_model

# ==========================================
# 1. LOAD MODEL
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'cnn_ocr_emnist_csv.h5')

if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(current_dir), 'cnn_ocr_emnist_csv.h5')

model = load_model(model_path)

label_map = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z',
    36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'
}

# ==========================================
# 2. FUNGSI PREDIKSI (Banyak Karakter)
# ==========================================
def process_and_predict(file_path):
    img = cv2.imread(file_path)
    if img is None: return None, 0, None
    
    canvas_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-process
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Cari semua kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "None", 0, canvas_img

    # Filter kontur yang terlalu kecil (noise) dan urutkan dari kiri ke kanan (x)
    char_list = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # Urutkan berdasarkan koordinat x (x, y, w, h)
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    full_text = ""
    total_conf = 0
    count = 0

    for (x, y, w, h) in sorted_boxes:
        # Filter: Abaikan jika objek terlalu kecil (misal: kurang dari 10px)
        if w < 10 or h < 10: 
            continue

        # Gambar kotak deteksi
        cv2.rectangle(canvas_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Potong & Padding
        roi = thresh[y:y+h, x:x+w]
        pad = max(w, h) // 4
        roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        # Resize & Normalisasi
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float32') / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        
        # Prediksi
        pred = model.predict(roi)
        idx = np.argmax(pred)
        conf = np.max(pred) * 100
        
        full_text += label_map[idx]
        total_conf += conf
        count += 1

    avg_conf = total_conf / count if count > 0 else 0
    return full_text, avg_conf, canvas_img

# ==========================================
# 3. GUI
# ==========================================
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        text, confidence, processed_img = process_and_predict(file_path)
        
        color_converted = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_converted)
        
        display_width = 500
        display_height = 400
        pil_img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(pil_img)
        lbl_canvas.configure(image=img_tk, text="")
        lbl_canvas.image = img_tk
        
        # Tampilkan teks hasil (bisa lebih dari 1 huruf)
        lbl_result.config(text=f"{text}")
        lbl_conf_text.config(text=f"Rata-rata Akurasi: {confidence:.1f}%")
        progress_acc['value'] = confidence

root = tk.Tk()
root.title("OCR Multi-Character View")
root.geometry("600x850")
root.configure(bg="#121212")

Label(root, text="SISTEM PREDIKSI OCR", font=("Arial", 20, "bold"), fg="#00E676", bg="#121212").pack(pady=20)

lbl_canvas = Label(root, text="BELUM ADA GAMBAR", bg="#1E1E1E", fg="#666", width=60, height=18)
lbl_canvas.pack(pady=10)

Button(root, text="UNGGAH FOTO", command=upload_image, bg="#00E676", fg="#121212", 
       font=("Arial", 12, "bold"), width=15, pady=10).pack(pady=10)

# Ukuran font disesuaikan agar jika teks panjang tidak meluber
lbl_result = Label(root, text="-", font=("Arial", 60, "bold"), bg="#121212", fg="white")
lbl_result.pack()

lbl_conf_text = Label(root, text="Akurasi: 0%", font=("Arial", 12), bg="#121212", fg="#AAA")
lbl_conf_text.pack(pady=5)

style = ttk.Style()
style.theme_use('default')
style.configure("green.Horizontal.TProgressbar", background='#00E676', troughcolor='#333', thickness=20)
progress_acc = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate", style="green.Horizontal.TProgressbar")
progress_acc.pack(pady=10)

root.mainloop()