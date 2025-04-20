from fake_news_detector import FakeNewsDetector

def main():
    # Inisialisasi detector
    detector = FakeNewsDetector()
    
    # Memuat model yang sudah dilatih sebelumnya
    print("Memuat model...")
    if detector.load_model('vectorizer_kaggle.pkl', 'classifier_kaggle.pkl'):
        print("Model berhasil dimuat!")
        
        while True:
            # Meminta input dari pengguna
            print("\n" + "-"*50)
            print("DETECTOR BERITA PALSU")
            print("-"*50)
            print("Masukkan berita yang ingin dideteksi (ketik 'exit' untuk keluar):")
            
            news_text = input("> ")
            
            if news_text.lower() == 'exit':
                print("Terima kasih telah menggunakan detector berita palsu!")
                break
            
            # Melakukan prediksi
            result = detector.predict(news_text)
            
            if result:
                # Menampilkan hasil dengan format yang lebih menarik
                print("\nHASIL DETEKSI:")
                print("-"*50)
                print(f"Status: {'PALSU ❌' if result['prediction'] == 'FAKE' else 'ASLI ✓'}")
                print(f"Tingkat Keyakinan: {result['confidence']:.2f}")
                
                # Menampilkan interpretasi
                if result['prediction'] == 'FAKE':
                    if result['confidence'] > 1.5:
                        print("Interpretasi: Sangat mungkin berita palsu.")
                    else:
                        print("Interpretasi: Kemungkinan berita palsu.")
                else:
                    if result['confidence'] > 1.5:
                        print("Interpretasi: Sangat mungkin berita asli.")
                    else:
                        print("Interpretasi: Kemungkinan berita asli.")
                
                print("-"*50)
            else:
                print("Error saat melakukan prediksi.")
    else:
        print("Gagal memuat model. Pastikan Anda telah melatih model terlebih dahulu.")

if __name__ == "__main__":
    main()