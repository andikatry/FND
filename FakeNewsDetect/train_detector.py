import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from fake_news_detector import FakeNewsDetector  # Mengimpor class yang sudah dibuat sebelumnya

def load_kaggle_dataset(true_news_path, fake_news_path):
    """
    Memuat dataset Kaggle Fake and Real News
    Parameters:
    - true_news_path: path ke file 'True.csv'
    - fake_news_path: path ke file 'Fake.csv'
    """
    try:
        # Memuat file CSV
        print("Memuat dataset berita nyata...")
        true_news = pd.read_csv(true_news_path)
        print("Memuat dataset berita palsu...")
        fake_news = pd.read_csv(fake_news_path)
        
        # Menambahkan label
        true_news['label'] = 0  # 0 untuk berita nyata
        fake_news['label'] = 1  # 1 untuk berita palsu
        
        # Menggabungkan data dan mengacaknya
        print("Menggabungkan dataset...")
        combined_data = pd.concat([true_news, fake_news], ignore_index=True)
        combined_data = shuffle(combined_data, random_state=42)
        
        # Mengambil kolom yang relevan saja
        if 'text' not in combined_data.columns and 'title' in combined_data.columns:
            # Dataset Kaggle ini memiliki kolom 'title' dan 'text' terpisah
            # Kita gabungkan keduanya menjadi satu kolom 'text'
            combined_data['text'] = combined_data['title'] + " " + combined_data['text']
        
        # Membuat dataset final dengan kolom yang diperlukan
        final_dataset = combined_data[['text', 'label']]
        
        print(f"Dataset berhasil dimuat. Total data: {len(final_dataset)}")
        print(f"Distribusi label: {final_dataset['label'].value_counts()}")
        
        return final_dataset
    
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        return None

def main():
    # Path ke file dataset Kaggle
    # Sesuaikan path ini dengan lokasi file Anda setelah mengunduh dan mengekstrak dataset
    true_news_path = "True.csv"  # Ganti dengan path sebenarnya ke file True.csv
    fake_news_path = "Fake.csv"  # Ganti dengan path sebenarnya ke file Fake.csv
    
    # Memuat dataset
    dataset = load_kaggle_dataset(true_news_path, fake_news_path)
    
    if dataset is None:
        print("Gagal memuat dataset. Program berhenti.")
        return
    
    # Inisialisasi detector
    detector = FakeNewsDetector()
    
    # Memecah dataset menjadi data yang lebih kecil jika dataset terlalu besar
    # Ini membantu jika Anda memiliki keterbatasan memori
    print("Melatih model...")
    max_samples = 20000  # Sesuaikan jumlah ini berdasarkan kapasitas komputer Anda
    
    if len(dataset) > max_samples:
        print(f"Dataset terlalu besar, menggunakan {max_samples} sampel secara acak...")
        dataset = dataset.sample(max_samples, random_state=42)
    
    # Melatih model
    results = detector.train(dataset, 'text', 'label')
    
    if results:
        print(f"Model berhasil dilatih dengan akurasi: {results['accuracy']:.2f}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")
        
        # Menyimpan model
        detector.save_model('vectorizer_kaggle.pkl', 'classifier_kaggle.pkl')
        print("Model berhasil disimpan.")
        
        # Contoh penggunaan model untuk prediksi
        print("\nMencoba model pada beberapa contoh berita:")
        
        test_news = [
            "President pledges support for new education initiatives across the country",
            "Scientists discover new species in Amazon rainforest that could revolutionize medicine",
            "BREAKING: Celebrity admits to faking their own death to escape fame",
            "Secret government program controlling weather patterns exposed by whistleblower",
            "Stock market showing signs of recovery after recent downturn",
            "Miracle cure discovered: This common fruit kills cancer cells in minutes"
        ]
        
        for news in test_news:
            result = detector.predict(news)
            print(f"\nBerita: {news}")
            print(f"Prediksi: {result['prediction']} (confidence: {result['confidence']:.2f})")

if __name__ == "__main__":
    main()