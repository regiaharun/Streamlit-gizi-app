import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Judul Aplikasi
st.title("Klasterisasi Data Gizi Menggunakan K-Means")

# Informasi Deskripsi
st.write("""
Aplikasi ini digunakan untuk melakukan klasterisasi pada data gizi individu 
berdasarkan parameter seperti berat badan, tinggi badan, dan nilai gizi.
""")

# Upload File Dataset
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load Dataset
    data = pd.read_csv(uploaded_file, header=None, names=["ID", "Nama", "Jenis Kelamin", "Usia", "Tinggi Badan", "Berat Badan", "Nilai Gizi"])
    st.subheader("Dataset yang Diupload:")
    st.dataframe(data)

    # Preprocessing Data
    data_cleaned = data.drop(columns=["ID", "Nama"])
    data_cleaned["Jenis Kelamin"] = data_cleaned["Jenis Kelamin"].map({"L": 1, "P": 0})
    data_cleaned = data_cleaned.apply(pd.to_numeric, errors="coerce").dropna()

    # Debug Data
    st.subheader("Data Setelah Preprocessing:")
    st.dataframe(data_cleaned)

    # Standardizing the dataset for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_cleaned)

    # Clustering with K-Means
    k = st.slider("Pilih Jumlah Klaster (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    data_cleaned["Cluster"] = kmeans.labels_

    # Display Clustering Results
    st.subheader("Hasil Klasterisasi:")
    st.dataframe(data_cleaned)

    # Visualisasi Klaster
    st.subheader("Visualisasi Klaster:")
    plt.figure(figsize=(8, 6))
    plt.scatter(data_cleaned["Tinggi Badan"], data_cleaned["Berat Badan"], c=data_cleaned["Cluster"], cmap="viridis", s=100)
    plt.title("Visualisasi Klaster Berdasarkan Tinggi dan Berat Badan")
    plt.xlabel("Tinggi Badan")
    plt.ylabel("Berat Badan")
    st.pyplot(plt)

    # Evaluasi Model
    sil_score = silhouette_score(data_scaled, kmeans.labels_)
    st.write(f"Silhouette Score: {sil_score:.2f}")
else:
    st.write("Silakan upload file dataset untuk memulai analisis.")
