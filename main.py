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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Home", "Classification", "Implementation"])

if menu == "Home":
    st.title("Informatika Pariwisata")
    st.markdown("# Judul Project")
    st.info("Analisis Sentimen Review Terhadap Pelayanan Hotel Jakarta menggunakan metode Random Forest dan Term Frequency-Inverse Document Frequency")
    st.markdown("# Dataset")
    st.info("Data yang digunakan pada laporan ini adalah data ulasan pelayanan hotel Jakarta. Data yang diambil dari Warung Amboina tersebut sebanyak lebih kurang 500 data dengan data yang diambil dalam waktu terdekat.")
    st.markdown("# Metode Usulan")
    st.info("Random Forest")

elif menu == "Classification":
    st.title("Classification")
    st.write("# Classification")
    st.info("## Random Forest")

    # Load dataset
    df = pd.read_csv("https://github.com/RibutDwiArtah023/AnalisisSentimenReview/raw/main/reviewHotelJakarta.csv")
    df['label'] = df['rating'].map({1.0:'Negatif', 2.0:'Negatif', 3.0:'Negatif', 4.0:'Positif', 5.0:'Positif'})
    sumdata = len(df)
    st.success(f"Total Data: {sumdata}")

    # Preprocessing
    st.write("## Preprocessing")
    st.write("### Text Cleaning")
    df['cleaned_review'] = df['review'].apply(cleaning)
    st.dataframe(df[['review', 'cleaned_review']].head(10))

    st.write("### Tokenization")
    df['tokens'] = df['cleaned_review'].apply(tokenize)
    st.dataframe(df[['cleaned_review', 'tokens']].head(10))

    st.write("### Stop Words Removal")
    df['cleaned_tokens'] = df['tokens'].apply(remove_stop_words)
    st.dataframe(df[['tokens', 'cleaned_tokens']].head(10))

    st.write("### Stemming")
    df['stemmed_tokens'] = df['cleaned_tokens'].apply(stem)
    st.dataframe(df[['cleaned_tokens', 'stemmed_tokens']].head(10))

    st.write("### TF-IDF Vectorization")
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['stemmed_tokens'].apply(lambda x: ' '.join(x)))
    st.dataframe(pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).head(10))

    st.write("### Dimensionality Reduction using PCA")
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X.toarray())
    st.dataframe(pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4']).head(10))

    # Classification
    st.write("# Classification")
    st.info("## Random Forest")
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X_pca, df['label'], test_size=0.3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### Model Accuracy")
    st.write(metrics.accuracy_score(y_test, y_pred))

    st.write("### Confusion Matrix")
    st.write(metrics.confusion_matrix(y_test, y_pred))

    st.write("### Classification Report")
    st.write(metrics.classification_report(y_test, y_pred))

elif menu == "Implementation":
    st.title("Implementation")
    st.info("## Predict Sentiment")
    user_input = st.text_area("Enter your review", "Type here...")
    cleaned_input = cleaning(user_input)
    tokens = tokenize(cleaned_input)
    no_stopwords = remove_stop_words(tokens)
    stemmed = stem(no_stopwords)
    processed_input = ' '.join(stemmed)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform([processed_input])
    
    # Dimensionality Reduction using PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X.toarray())
    
    # Load the trained model
    model = RandomForestClassifier()  # Load your trained model here

    # Predict using the loaded model
    prediction = model.predict(X_pca)

    if prediction == 'Positif':
        st.success("The review sentiment is Positive")
    else:
        st.error("The review sentiment is Negative")
