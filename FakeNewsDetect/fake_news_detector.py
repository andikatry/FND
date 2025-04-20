import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.classifier = PassiveAggressiveClassifier(max_iter=50)
        self.model_trained = False
        
    def clean_text(self, text):
        """Membersihkan teks dari karakter khusus"""
        if isinstance(text, str):
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()
        return ""
        
    def load_data(self, filepath):
        """Memuat dataset dari file CSV"""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def train(self, df, text_column, label_column):
        """Melatih model menggunakan dataset yang diberikan"""
        try:
            # Mempersiapkan data
            df[text_column] = df[text_column].apply(self.clean_text)
            
            # Memisahkan data menjadi fitur dan label
            X = df[text_column]
            y = df[label_column]
            
            # Membagi data menjadi training dan testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Vectorizing fitur teks
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            X_test_vectorized = self.vectorizer.transform(X_test)
            
            # Melatih classifier
            self.classifier.fit(X_train_vectorized, y_train)
            
            # Evaluasi model
            y_pred = self.classifier.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)
            
            self.model_trained = True
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': confusion,
                'test_data': (X_test, y_test)
            }
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def predict(self, text):
        """Memprediksi apakah teks merupakan berita palsu atau tidak"""
        if not self.model_trained:
            return "Model belum dilatih"
        
        try:
            cleaned_text = self.clean_text(text)
            text_vectorized = self.vectorizer.transform([cleaned_text])
            prediction = self.classifier.predict(text_vectorized)[0]
            confidence = np.max(self.classifier.decision_function(text_vectorized))
            
            result = "REAL" if prediction == 0 else "FAKE"
            return {
                'prediction': result,
                'confidence': abs(confidence)
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def save_model(self, vectorizer_path, classifier_path):
        """Menyimpan model yang sudah dilatih"""
        if not self.model_trained:
            return "Model belum dilatih"
        
        import pickle
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, vectorizer_path, classifier_path):
        """Memuat model yang sudah dilatih sebelumnya"""
        import pickle
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.model_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False