import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# LOAD DATA
df = pd.read_excel("DATASET JUDUL SKRIPSI fiks.xlsx")

X = df.iloc[:, 0].astype(str)
y = df.iloc[:, 1]

# TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# ANN (INI REAL ANN)
model = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    activation='relu',
    max_iter=500,
    random_state=42
)

model.fit(X_tfidf, y)

# STREAMLIT UI
st.title("Prediksi Judul Skripsi")

judul = st.text_area("Masukkan Judul")

if st.button("Prediksi"):
    if judul.strip() == "":
        st.warning("Isi dulu judulnya")
    else:
        pred = model.predict(vectorizer.transform([judul]))[0]

        if pred == 1 or pred == "STEM":
            st.success("STEM")
        else:
            st.error("NON STEM")
