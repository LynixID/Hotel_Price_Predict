# Import library yang diperlukan
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error

# Konfigurasi tampilan halaman
st.set_page_config(layout="wide", page_title="Prediksi Harga Hotel")

# Fungsi untuk memuat dan membersihkan data
def muat_data():
    data = pd.read_csv("booking_hotel.csv", encoding="latin1")
    data.columns = data.columns.str.strip()
    data['Room Price'] = data['Room Price (in BDT or any other currency)'].str.replace("[^\d]", "", regex=True).astype(float)
    return data.dropna(subset=['Room Price'])

# Memuat dataset
dataset = muat_data()

# Konfigurasi sidebar
st.sidebar.image("image.jpg", width=200)  # Width tetap bisa digunakan untuk sidebar
st.sidebar.title("🏨 Menu Navigasi")
st.sidebar.markdown("---")

# Menu sidebar
selected = st.sidebar.selectbox(
    "Pilih Menu",
    ["Beranda", "Dataset", "Visualisasi", "Prediksi"],
)

# Informasi di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Informasi Aplikasi")
st.sidebar.info("Aplikasi ini menggunakan machine learning untuk memprediksi harga kamar hotel.")
st.sidebar.markdown("### 📊 Statistik Data")
st.sidebar.metric("Total Data", f"{len(dataset):,} baris")

# Fungsi untuk halaman berandaS
def beranda():
    st.title("🏢 Aplikasi Prediksi Harga Hotel")
    st.markdown("---")
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### Selamat Datang di Aplikasi Prediksi Harga Hotel!
        
        Aplikasi ini membantu memperkirakan harga kamar hotel berdasarkan:
        - 📍 Lokasi Hotel
        - 🛏️ Jenis Kamar
        - 🛋️ Tipe Tempat Tidur
        """)
        
        st.info("""
        ### 🎯 Fitur Utama:
        1. **Dataset**: Melihat data hotel yang tersedia
        2. **Visualisasi**: Analisis visual data harga hotel
        3. **Prediksi**: Prediksi harga berdasarkan preferensi
        """)
    with col2:
        with col2:
            st.image("image.jpg", caption="Prediksi Harga Hotel", use_container_width=True)

# Fungsi untuk halaman dataset
def dataset_view():
    st.title("📊 Dataset Hotel")
    st.markdown("---")
    
    col1, col2 = st.columns([3,1])
    with col1:
        st.dataframe(dataset, use_container_width=True)
    with col2:
        st.metric("Jumlah Data", f"{dataset.shape[0]:,}")
        st.metric("Jumlah Kolom", f"{dataset.shape[1]}")
        
        st.markdown("### 📈 Ringkasan Statistik")
        st.write("Harga Terendah:", f"Rp {dataset['Room Price'].min():,.2f}")
        st.write("Harga Tertinggi:", f"Rp {dataset['Room Price'].max():,.2f}")
        st.write("Harga Rata-rata:", f"Rp {dataset['Room Price'].mean():,.2f}")

# Fungsi untuk halaman visualisasi
def visualisasi():
    st.title("📈 Visualisasi Data")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Harga", "📍 Harga per Lokasi", "🛏️ Analisis Kamar", "⭐ Rating Hotel"])
    
    with tab1:
        st.markdown("### Distribusi Harga Kamar Hotel")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(dataset["Room Price"], kde=True, ax=ax1)
        ax1.set_title("Distribusi Harga Kamar")
        st.pyplot(fig1)
        
    with tab2:
        st.markdown("### Perbandingan Harga Berdasarkan Lokasi")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=dataset, x="Location", y="Room Price", ax=ax2)
        plt.xticks(rotation=45)
        ax2.set_title("Harga Kamar Berdasarkan Lokasi")
        st.pyplot(fig2)
    
    with tab3:
        st.markdown("### Analisis Harga Berdasarkan Tipe Kamar dan Tempat Tidur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_room_price = dataset.groupby("Room Type")["Room Price"].mean().sort_values(ascending=True)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            avg_room_price.plot(kind='barh', ax=ax3)
            ax3.set_title("Rata-rata Harga per Jenis Kamar")
            ax3.set_xlabel("Harga (Rp)")
            plt.tight_layout()
            st.pyplot(fig3)
        
        with col2:
            avg_bed_price = dataset.groupby("Bed Type")["Room Price"].mean().sort_values(ascending=True)
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            avg_bed_price.plot(kind='barh', ax=ax4)
            ax4.set_title("Rata-rata Harga per Tipe Tempat Tidur")
            ax4.set_xlabel("Harga (Rp)")
            plt.tight_layout()
            st.pyplot(fig4)
        
        st.markdown("### 📊 Rangkuman Statistik")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Harga Tertinggi", f"Rp {dataset['Room Price'].max():,.2f}")
        with col4:
            st.metric("Harga Terendah", f"Rp {dataset['Room Price'].min():,.2f}")
        with col5:
            st.metric("Harga Rata-rata", f"Rp {dataset['Room Price'].mean():,.2f}")
    
    with tab4:
        st.markdown("### ⭐ Line Chart Rating Hotel")
        if "Rating" in dataset.columns:  # Pastikan kolom Rating ada
            sorted_dataset = dataset.sort_values(by="Hotel Name")  # Urutkan berdasarkan nama hotel
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=sorted_dataset, x="Hotel Name", y="Rating", ax=ax5, marker="o")
            ax5.set_title("Rating Hotel")
            ax5.set_xlabel("Nama Hotel")
            ax5.set_ylabel("Rating")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig5)
        else:
            st.error("Kolom 'Rating' tidak ditemukan dalam dataset!")

# Fungsi untuk halaman prediksi
def prediksi():
    st.title("🔮 Prediksi Harga Hotel")
    st.markdown("---")
    
    # Penjelasan cara kerja prediksi
    st.markdown("""
    ### ℹ️ Cara Kerja Prediksi:
    1. Model mempelajari pola dari dataset yang berisi data historis harga hotel
    2. Prediksi dilakukan berdasarkan 3 faktor utama:
        - Lokasi hotel yang dipilih
        - Jenis kamar yang diinginkan
        - Tipe tempat tidur yang tersedia
    3. Model akan menganalisis data historis untuk menemukan harga yang paling sesuai
    """)
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.markdown("### Parameter Input")
        lokasi = st.selectbox("📍 Lokasi", dataset["Location"].unique())
        jenis_kamar = st.selectbox("🛏️ Jenis Kamar", dataset["Room Type"].unique())
        jenis_tempat_tidur = st.selectbox("🛋️ Jenis Tempat Tidur", dataset["Bed Type"].unique())
        
    with col2:
        st.markdown("### Hasil Prediksi")
        
        X = dataset[["Location", "Room Type", "Bed Type"]]
        X = pd.get_dummies(X, drop_first=True)
        y = dataset["Room Price"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        data_input = pd.DataFrame([[lokasi, jenis_kamar, jenis_tempat_tidur]], 
                                 columns=["Location", "Room Type", "Bed Type"])
        data_input = pd.get_dummies(data_input)
        data_input = data_input.reindex(columns=X.columns, fill_value=0)
        hasil_prediksi = model.predict(data_input)
        
        st.metric("💰 Prediksi Harga", f"Rp {hasil_prediksi[0]:,.2f}")
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.metric("📉 Tingkat Akurasi Prediksi", f"±Rp {mae:,.2f}")
        
        st.info("""
        💡 Informasi Tingkat Akurasi:
        - Angka ini menunjukkan rata-rata selisih antara harga prediksi dengan harga sebenarnya
        - Contoh: Jika tingkat akurasi ±Rp 100.000, prediksi harga bisa lebih tinggi atau lebih rendah sekitar Rp 100.000
        - Semakin kecil nilai ini, semakin akurat prediksi yang dihasilkan
        """)

# Router halaman
if selected == "Beranda":
    beranda()
elif selected == "Dataset":
    dataset_view()
elif selected == "Visualisasi":
    visualisasi()
elif selected == "Prediksi":
    prediksi()