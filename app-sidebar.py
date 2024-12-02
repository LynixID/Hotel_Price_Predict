import pandas as pd
import streamlit as st
import base64

# Fungsi untuk menghitung skor berdasarkan kategori
def calculate_scores(df, answers):
    scores = df[['cat']].copy()
    scores['score'] = answers
    total_scores = scores.groupby('cat')['score'].sum().to_dict()
    return total_scores

# Fungsi untuk mengunduh dataset
def download_csv(data, filename, label):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # encode to Base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

# Membuat aplikasi Streamlit
def main():
    # Load dataset
    file_path = "DASS21_v1.csv"  # Sesuaikan dengan lokasi dataset Anda
    df = pd.read_csv(file_path)

    # Sidebar untuk navigasi
    st.sidebar.title("📋 **DASS-21 Assessment**")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigasi Utama")
    menu = st.sidebar.selectbox(
        "Pilih Halaman",
        ["🏠 Home", "📊 Dataset", "🧠 Prediksi"]
    )

    # Sidebar informasi tambahan
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tentang")
    st.sidebar.info(
        """
        Aplikasi ini didasarkan pada **DASS-21**, 
        yang membantu mengukur tingkat **Stress**, **Anxiety**, dan **Depression**.
        """
    )

    # Home Page
    if menu == "🏠 Home":
        st.title("🏠 Aplikasi Prediksi Tingkat Anxiety DASS-21")
        col1, col2 = st.columns([2, 1])  # Membagi menjadi 2 kolom
        with col1:
            st.markdown("""
            ### Selamat Datang!
            Aplikasi ini dirancang untuk membantu Anda menilai tingkat **Stress**, **Anxiety**, dan **Depression** berdasarkan kuesioner **DASS-21**.

            #### Fitur Utama:
            - **Kuesioner Interaktif**: Jawab 21 pertanyaan dengan mudah.
            - **Hasil Klasifikasi**: Analisis berdasarkan kategori.
            - **Akses Dataset**: Tinjau atau unduh data kuesioner.
            
            > Gunakan aplikasi ini untuk memahami kesehatan mental Anda dengan lebih baik.
            """, unsafe_allow_html=True)
        with col2:
            st.image("anxiety.jpg", caption="DASS-21 Assessment", use_column_width=True)

    # Dataset Page
    elif menu == "📊 Dataset":
        st.title("📊 Dataset DASS-21")
        st.markdown("""
        Berikut adalah dataset dari kuesioner **DASS-21**.
        Anda juga dapat mengunduh dataset ini untuk kebutuhan analisis lebih lanjut.
        """)

        # Tampilkan dataset
        st.dataframe(df.style.set_properties(**{'background-color': '#f9f9f9', 
                                                'color': '#333333', 
                                                'border': '1px solid #dddddd'}))

        # Tambahkan tautan unduh
        st.markdown(download_csv(df, "DASS21_v1.csv", "📥 Unduh Dataset"), unsafe_allow_html=True)

    # Prediksi Page
    elif menu == "🧠 Prediksi":
        st.title("🧠 Prediksi Tingkat Anxiety dengan DASS-21")
        st.write("Jawab pertanyaan berikut sesuai dengan kondisi Anda:")
        user_answers = []

        # Input jawaban untuk setiap pertanyaan
        for index, row in df.iterrows():
            answer = st.radio(
                f"{row['qno']}. {row['qtext']}",
                options=[0, 1, 2, 3],
                index=0,
                horizontal=True
            )
            user_answers.append(answer)

        # Prediksi skor
        if st.button("💡 Lihat Hasil"):
            total_scores = calculate_scores(df, user_answers)
            st.success("Berikut adalah hasil Anda:")
            for category, score in total_scores.items():
                st.write(f"**{category.capitalize()}**: {score}")

            # Klasifikasi berdasarkan skor (contoh threshold sederhana)
            st.write("### Kategori Hasil:")
            thresholds = {"s": 14, "a": 7, "d": 10}
            for category, score in total_scores.items():
                level = (
                    "Ringan"
                    if score <= thresholds[category]
                    else "Sedang"
                    if score <= thresholds[category] * 2
                    else "Berat"
                )
                st.write(f"- **{category.capitalize()}**: {level}")

if __name__ == "__main__":
    main()
