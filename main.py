import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from itertools import chain
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Download NLTK resources
nltk.download('popular')
nltk.download('stopwords')

# Function to clean text
def cleaning(text):
    # HTML Tag Removal
    text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))

    # Case folding
    text = text.lower()

    # Trim text
    text = text.strip()

    # Remove punctuations, special characters, and double spaces
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)

    # Number removal
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Change 'nan' text to whitespace for removal later
    text = re.sub('nan', '', text)

    return text

# Function to tokenize text
def tokenize(text):
    return word_tokenize(text)

# Function to remove stop words
def remove_stop_words(tokens):
    stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
    return [w for w in tokens if not w in stop_words]

# Function to stem tokens
def stem(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(w) for w in tokens]

# Set Streamlit page configuration
st.set_page_config(page_title="Informatika Pariwisata", page_icon='')

with st.container():
    with st.sidebar:
        choose = st.selectbox("Menu", ["Home", "Implementation"])

    if choose == "Home":
        st.markdown('<h1 style="text-align: center;">Informatika Pariwisata</h1>', unsafe_allow_html=True)
        st.markdown("# Judul Project")
        st.info("Analisis Sentimen Review Terhadap Pelayanan Hotel Jakarta menggunakan metode Random Forest dan Term Frequency-Inverse Document Frequency")
        st.markdown("# Dataset")
        st.info("Data yang digunakan pada laporan ini adalah data ulasan pelayanan hotel Jakarta. Data yang diambil dari Warung Amboina tersebut sebanyak lebih kurang 500 data dengan data yang diambil dalam waktu terdekat.")
        st.markdown("# Metode Usulan")
        st.info("Random Forest")

    elif choose == "Implementation":
        st.title("Informatika Pariwisata - Ribut Dwi Artah (200411100023)")
        desc, dataset, preprocessing, classification, implementation = st.columns(5)

        with desc:
            st.write("# About Dataset")
            st.write("## Content")
            st.markdown("1. **Name**: Nama pengguna yang memberikan komentar di Warung Amboina.")
            st.markdown("2. **Text**: Komentar yang diberikan oleh pengguna.")
            st.markdown("3. **Label**: Label positif dan negatif dari cita rasa makanan di Warung Amboina.")
            st.markdown("4. **Review URL**: Link halaman ulasan di Google Maps untuk Warung Amboina.")
            st.markdown("5. **Reviewer URL**: Link profil pengguna yang menambahkan ulasan di Google Maps untuk Warung Amboina.")
            st.markdown("6. **Stars**: Bintang yang diberikan oleh pengguna saat mengulas di Google Maps untuk Warung Amboina.")
            st.markdown("7. **Publish at**: Waktu pengguna menambahkan ulasan di Google Maps untuk Warung Amboina.")

            st.write("## Repository Github")
            st.markdown("Klik link di bawah ini untuk mengakses kode sumber:")
            st.markdown("[Link Repository Github](https://github.com/RibutDwiArtah023/AnalisisSentimenReview)")

        with dataset:
            st.write("# Load Dataset")
            df = pd.read_csv("https://github.com/RibutDwiArtah023/AnalisisSentimenReview/raw/main/reviewHotelJakarta.csv")
            df['label'] = df['rating'].map({1.0:'Negatif', 2.0:'Negatif', 3.0:'Negatif', 4.0:'Positif', 5.0:'Positif'})
            sumdata = len(df)
            st.success(f"Total Data: {sumdata}")

            st.write("## Dataset Explanation")
            st.info("Classes:")
            st.markdown("- Positif")
            st.markdown("- Negatif")

            col1, col2 = st.columns(2)
            with col1:
                st.info("Data Types")
                st.write(df.dtypes)
            with col2:
                st.info("Empty Data")
                st.write(df.isnull().sum())

        with preprocessing:
            st.write("# Preprocessing")
            st.info("## Text Cleaning")

            # Clean text
            df['cleaned_review'] = df['review'].apply(cleaning)
            st.write("### Cleaned Reviews")
            st.dataframe(df[['review', 'cleaned_review']].head(10))

            st.info("## Tokenization")
            # Tokenize text
            df['tokens'] = df['cleaned_review'].apply(tokenize)
            st.write("### Tokenized Reviews")
            st.dataframe(df[['cleaned_review', 'tokens']].head(10))

            st.info("## Stop Words Removal")
            # Remove stop words
            df['cleaned_tokens'] = df['tokens'].apply(remove_stop_words)
            st.write("### Stop Words Removed")
            st.dataframe(df[['tokens', 'cleaned_tokens']].head(10))

            st.info("## Stemming")
            # Stemming
            df['stemmed_tokens'] = df['cleaned_tokens'].apply(stem)
            st.write("### Stemmed Tokens")
            st.dataframe(df[['cleaned_tokens', 'stemmed_tokens']].head(10))

            st.info("## TF-IDF Vectorization")
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer()
            X = tfidf_vectorizer.fit_transform(df['stemmed_tokens'].apply(lambda x: ' '.join(x)))
            st.write("### TF-IDF Matrix")
            st.dataframe(pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).head(10))

            st.info("## Dimensionality Reduction using PCA")
            # PCA
            pca = PCA(n_components=4)
            X_pca = pca.fit_transform(X.toarray())
            st.write("### PCA Components")
            st.dataframe(pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4']).head(10))

        with classification:
            st.write("# Classification")
            st.info("## Random Forest")
            # Classification with Random Forest
            model = RandomForestClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X_pca, df['label'], test_size=0.3, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.info("### Model Accuracy")
            st.write(metrics.accuracy_score(y_test, y_pred))

            st.info("### Confusion Matrix")
            st.write(metrics.confusion_matrix(y_test, y_pred))

            st.info("### Classification Report")
            st.write(metrics.classification_report(y_test, y_pred))

        with implementation:
            st.write("# Implementation")
            st.info("## Predict Sentiment")
            user_input = st.text_area("Enter your review", "Type here...")
            cleaned_input = cleaning(user_input)
            tokens = tokenize(cleaned_input)
            no_stopwords = remove_stop_words(tokens)
            stemmed = stem(no_stopwords)
            processed_input = ' '.join(stemmed)
            input_vector = tfidf_vectorizer.transform([processed_input])
            input_pca = pca.transform(input_vector.toarray())
            prediction = model.predict(input_pca)

            if prediction == 'Positif':
                st.success("The review sentiment is Positive")
            else:
                st.error("The review sentiment is Negative")
