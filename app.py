import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =====================
# LOAD DATASET
# =====================
df = pd.read_excel("DATASET JUDUL SKRIPSI fiks.xlsx")

# Ambil kolom pertama = judul
# Ambil kolom kedua = label
X = df.iloc[:, 0].astype(str)
y = df.iloc[:, 1]

# =====================
# TRAIN MODEL
# =====================
vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# =====================
# STREAMLIT
# =====================
st.title("Prediksi Kategori Judul Skripsi")

st.write("Masukkan judul skripsi untuk mengetahui kategori STEM atau NON STEM")

judul = st.text_area("Masukkan Judul Skripsi")

if st.button("Prediksi"):

    if judul.strip() == "":
        st.warning("Silakan masukkan judul skripsi")
    else:

        data = vectorizer.transform([judul])

        hasil = model.predict(data)[0]

        st.subheader("Hasil Prediksi")

        # SESUAIKAN LABEL DI DATASET
        if hasil == 0:
            st.success("STEM")
        else:
            st.error("NON STEM")