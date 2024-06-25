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
from streamlit_option_menu import option_menu
st.set_page_config(page_title="Informatika Pariwisata", page_icon='')

# Function for text cleaning
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

    # Replace 'nan' with whitespace to be removed later
    text = re.sub('nan', '', text)

    return text

# Function for tokenization
def tokenize(text):
    return word_tokenize(text)

# Function for stop words removal
stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
def remove_stop_words(tokens):
    return [w for w in tokens if not w in stop_words]

# Function for stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem(tokens):
    return [stemmer.stem(w) for w in tokens]

with st.container():
    with st.sidebar:
        choose = option_menu("Menu", ["Home", "Implementation"],
                             icons=['house', 'basket-fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "#fb6f92", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#c6e2e9"},
            "nav-link-selected": {"background-color": "#a7bed3"},
        }
        )

    if choose == "Home":
        
        st.markdown('<h1 style = "text-align: center;"> <b>Informatika Pariwisata</b> </h1>', unsafe_allow_html = True)
        st.markdown('')
        st.markdown("# Judul Project ")
        st.info("Analisis Sentimen Review Terhadap Pelayanan Hotel Jakarta menggunakan metode Random Forest dan Term Frequency-Inverse Document Frequency")
        st.markdown("# Dataset ")
        st.info("Data yang digunakan pada laporan ini adalah data ulasan pelayanan hotel Jakarta. Data yang diambil dari Warung Amboina tersebut sebanyak lebih kurang 500 data dengan data yang diambil dalam waktu terdekat.")
        st.markdown("# Metode Usulan ")
        st.info("Random Forest")
        
    elif choose == "Implementation":
        st.title("Informatika Pariwisata")
        st.write("Ribut Dwi Artah - 200411100023")
        desc, dataset, preprocessing, classification, implementation = st.tabs(["Deskripsi Data", "Dataset", "Preprocessing", "Classification", "Implementation"])
        with desc:
            st.write("# About Dataset")
            
            st.write("## Content")
            st.write("""
            1.  Name :
                > Tabel Name berisi nama pengguna yang memberikan komentar di Warung Amboina.
            2.  Text :
                > Tabel Text berisi komentar yang diberikan oleh pengguna.
            3.  Label :
                > Tabel Label berisi label Positif dan Negatif dari cita rasa makanan di Warung Amboina.
            4. Review URL :
                > Tabel Review URL berisi link yang mengarahkan ke halaman ulasan yang ada di Google Maps pada Warung Amboina.
            5. Reviewer URL :
                > Tabel Reviewer URL berisi link yang mengarahkan ke profil pengguna yang menambahkan ulasan yang ada di Google Maps pada Warung Amboina.
            6. Stars :
                > Tabel Stars berisi bintang yang diberikan oleh pengguna saat mengulas Warung Amboina di Google Maps.
            7. Publish at :
                > Tabel Publish at berisi waktu pengguna menambahkan ulasan di Warung Amboina pada Google Maps.
                    """)

            st.write("## Repository Github")
            st.write(" Click the link below to access the source code")
            repo = "https://github.com/RibutDwiArtah023/AnalisisSentimenReview"
            st.markdown(f'[ Link Repository Github ]({repo})')
        with dataset:
            st.write("""# Load Dataset""")
            df = pd.read_csv("https://github.com/RibutDwiArtah023/AnalisisSentimenReview/raw/main/reviewHotelJakarta.csv")
            df['label'] = df['rating'].map({1.0:'Negatif', 2.0:'Negatif', 3.0:'Negatif', 4.0:'Positif', 5.0:'Positif'})
            df
            sumdata = len(df)
            st.success(f"#### Total Data : {sumdata}")
            st.write("## Dataset Explanation")
            st.info("#### Classes :")
            st.write("""
            1. Positif
            2. Negatif
            """)

            col1,col2 = st.columns(2)
            with col1:
                st.info("#### Data Type")
                df.dtypes
            with col2:
                st.info("#### Empty Data")
                st.write(df.isnull().sum())
                #===================================
             
                
                
        with preprocessing : 
            st.write("""# Preprocessing""")
            st.write("""
            > Preprocessing data adalah proses menyiapkan data mentah dan membuatnya cocok untuk model pembelajaran mesin. Ini adalah langkah pertama dan penting saat membuat model pembelajaran mesin. Saat membuat proyek pembelajaran mesin, kami tidak selalu menemukan data yang bersih dan terformat.
            """)
            st.info("## Cleaned Data")
            
            # Apply cleaning, tokenization, stop words removal, and stemming
            df['cleaned_text'] = df['review'].apply(cleaning)
            df['review_tokens'] = df['cleaned_text'].apply(tokenize)
            df['review_tokens'] = df['review_tokens'].apply(remove_stop_words)
            df['review_tokens'] = df['review_tokens'].apply(stem)

            Sumdata = len(df)
            st.success(f"#### Total Cleaned Data : {Sumdata}")
            
            st.info("## TF - IDF (Term Frequency Inverse Document Frequency)")
            from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
            countvectorizer = CountVectorizer()
            tfidfvectorizer = TfidfVectorizer()
            tfidf = TfidfVectorizer()
            countwm = CountVectorizer()
            documents_list = df['review_tokens'].apply(lambda x: ' '.join(x)).tolist()
            count_wm = countwm.fit_transform(documents_list)
            train_data = tfidf.fit_transform(documents_list)
            count_array = count_wm.toarray()
            tf_idf_array = train_data.toarray()
            words_set = tfidf.get_feature_names_out()
            count_set = countwm.get_feature_names_out()
            df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
            df_tf_idf
            
            st.info("## Dimension Reduction using PCA")
            # Impor library yang dibutuhkan
            from sklearn.decomposition import PCA
            # Inisialisasi objek PCA dengan 4 komponen
            pca = PCA(n_components=4)
            # Melakukan fit transform pada data
            X_pca = pca.fit_transform(df_tf_idf)
            X_pca.shape
            
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder() 
            df['label']= label_encoder.fit_transform(df['label'])

            y = df['label'].values
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y ,test_size = 0.7, random_state =1)


        with classification : 
            st.write("""# Classification""")
            st.info("## Random Forest")
            st.write(""" > Random Forest adalah algoritma machine learning yang menggabungkan keluaran dari beberapa decision tree untuk mencapai satu hasil. Random Forest bekerja dengan membangun beberapa decision tree dan menggabungkannya demi mend
