import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding="latin1")
    data.columns = data.columns.str.strip()  # Clean up column names
    data['Room Price'] = data['Room Price (in BDT or any other currency)'].str.replace("[^\d]", "", regex=True).astype(float)
    data = data.dropna(subset=['Room Price'])  # Remove rows with NaN in Room Price
    return data


dataset = load_data("booking_hotel.csv")  # Ganti dengan path file yang sesuai

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Pilih Halaman", ["Homepage", "Dataset", "Visualization", "Prediction"])

if page == "Homepage":
    st.title("Hotel Price Prediction App")
    st.write("""
    Aplikasi ini memprediksi harga kamar hotel berdasarkan parameter seperti lokasi, jenis kamar, jenis tempat tidur, dan lain-lain.
    Anda dapat menjelajahi dataset, melihat visualisasi, dan melakukan prediksi harga!
    """)
    st.image("image.jpg", caption="Prediksi Harga Hotel")

elif page == "Dataset":
    st.title("Dataset")
    st.write("Berikut adalah data yang digunakan untuk model:")
    st.write(dataset)
    st.write("Jumlah data:", dataset.shape)

elif page == "Visualization":
    st.title("Visualizations")
    st.write("Analisis visual data.")
    
    # Distribusi Harga Kamar
    st.write("Distribusi Harga Kamar")
    fig, ax = plt.subplots()
    sns.histplot(dataset["Room Price"], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Harga Kamar Berdasarkan Lokasi
    st.write("Harga Kamar Berdasarkan Lokasi")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=dataset, x="Location", y="Room Price", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Prediction":
    st.title("Prediction")
    st.write("Masukkan parameter untuk memprediksi harga kamar hotel.")
    
    # Input features for prediction
    location = st.selectbox("Lokasi", dataset["Location"].unique())
    room_type = st.selectbox("Jenis Kamar", dataset["Room Type"].unique())
    bed_type = st.selectbox("Jenis Tempat Tidur", dataset["Bed Type"].unique())

    # Tombol trigger untuk prediksi
    if st.button("Prediksi Harga"):
        # Prepare the data for prediction
        X = dataset[["Location", "Room Type", "Bed Type"]]
        X = pd.get_dummies(X, drop_first=True)
        y = dataset["Room Price"]

        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make prediction
        input_data = pd.DataFrame([[location, room_type, bed_type]], 
                                  columns=["Location", "Room Type", "Bed Type"])
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_data)

        st.write(f"Prediksi Harga Kamar: ${prediction[0]:,.2f}")
