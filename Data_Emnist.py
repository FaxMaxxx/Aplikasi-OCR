from emnist import extract_training_samples, extract_test_samples

print("Sedang mendownload dan memuat data EMNIST (Split: Balanced)...")

# 'balanced' adalah split yang disarankan karena ukurannya lebih kecil dari 'byclass'
# tapi tetap mencakup Angka (0-9), Huruf Besar, dan Huruf Kecil.
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

print(f"Download selesai!")
print(f"Jumlah Data Train: {X_train.shape}")
print(f"Jumlah Data Test: {X_test.shape}")