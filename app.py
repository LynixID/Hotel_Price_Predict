import pandas as pd
import streamlit as st
import base64

# Fungsi untuk menghitung skor berdasarkan kategori
def calculate_scores(df, answers):
    scores = df[['cat']].copy()
    scores['score'] = answers
    total_scores = scores.groupby('cat')['score'].sum().to_dict()
    return total_scores

# Membuat aplikasi Streamlit
def main():
    # Load dataset
    file_path = "DASS21_v1.csv"  # Sesuaikan dengan lokasi dataset Anda
    df = pd.read_csv(file_path)

    # Sidebar untuk navigasi
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Pilih Halaman", ["Home", "Dataset", "Prediksi"])

    # Home Page
    if menu == "Home":
        st.title("Aplikasi Prediksi Tingkat Anxiety DASS-21")
        st.image("https://via.placeholder.com/800x400.png?text=DASS-21", caption="DASS-21 Assessment", use_column_width=True)
        st.markdown("""
        ### Tentang Aplikasi
        Aplikasi ini mengukur tingkat **Stress**, **Anxiety**, dan **Depression** berdasarkan kuesioner **DASS-21**.
        
        ### Cara Kerja
        1. Jawab 21 pertanyaan yang tersedia pada menu **Prediksi**.
        2. Aplikasi akan menghitung skor dan memberikan hasil dalam kategori:
           - Ringan
           - Sedang
           - Berat
        
        """)

    # Dataset Page
    elif menu == "Dataset":
        st.title("Dataset DASS-21")
        st.write("Berikut adalah data dari kuesioner DASS-21:")
        st.dataframe(df)

    # Prediksi Page
    elif menu == "Prediksi":
        st.title("Prediksi Tingkat Anxiety dengan DASS-21")
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
        if st.button("Lihat Hasil"):
            total_scores = calculate_scores(df, user_answers)
            st.success("Berikut adalah hasil Anda:")
            for category, score in total_scores.items():
                st.write(f"**{category.capitalize()}**: {score}")

            # Klasifikasi berdasarkan skor (contoh threshold sederhana)
            st.write("### Kategori Hasil:")
            thresholds = {"s": 14, "a": 7, "d": 10}
            for category, score in total_scores.items():
                level = "Ringan" if score <= thresholds[category] else "Sedang" if score <= thresholds[category] * 2 else "Berat"
                st.write(f"- **{category.capitalize()}**: {level}")

if __name__ == "__main__":
    main()
