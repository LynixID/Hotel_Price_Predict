import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64

# Fungsi untuk menghitung skor berdasarkan kategori yang diberikan dalam jawaban
def calculate_scores(df, answers):
    # Membuat salinan dari DataFrame untuk kategori dan jawaban
    scores = df[['cat']].copy()
    scores['score'] = answers  # Menambahkan kolom 'score' yang berisi jawaban
    # Mengelompokkan skor berdasarkan kategori dan menghitung total skor per kategori
    total_scores = scores.groupby('cat')['score'].sum().to_dict()
    return total_scores

# Fungsi untuk membuat grafik batang (Bar Chart) berdasarkan jawaban
def plot_bar_chart(df, answered_count, user_answers):
    # Menghitung jumlah jawaban per kategori (0-3)
    category_counts = pd.Series(user_answers).value_counts()

    # Membuat figure dan axes untuk grafik
    fig, ax = plt.subplots()
    # Menggambar grafik batang
    category_counts.plot(kind='bar', color=['#6baed6', '#fd8d3c', '#74c476'], ax=ax)
    # Memberikan judul dan label sumbu X dan Y
    ax.set_title(f"Distribusi Jawaban ({answered_count} Pertanyaan Dijawab)")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Jumlah Jawaban")
    
    # Menampilkan nama kategori pada sumbu X
    categories = ['Stress (S)', 'Anxiety (A)', 'Depression (D)']
    present_categories = [categories[i] for i in range(len(categories)) if i in category_counts.index]
    
    ax.set_xticks(range(len(present_categories)))
    ax.set_xticklabels(present_categories)
    ax.bar_label(ax.containers[0])  # Menambahkan label jumlah di atas batang
    return fig

# Fungsi untuk membuat grafik garis (Line Chart) berdasarkan jawaban
def plot_line_chart(df, answered_count, user_answers):
    # Menghitung jumlah jawaban per kategori
    category_counts = pd.Series(user_answers).value_counts()

    # Membuat figure dan axes untuk grafik
    fig, ax = plt.subplots()
    # Menggambar grafik garis
    category_counts.sort_index().plot(kind='line', marker='o', color='b', ax=ax)
    ax.set_title(f"Distribusi Jawaban ({answered_count} Pertanyaan Dijawab)")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Jumlah Jawaban")

    # Menampilkan nama kategori pada sumbu X
    categories = ['Stress (S)', 'Anxiety (A)', 'Depression (D)']
    present_categories = [categories[i] for i in range(len(categories)) if i in category_counts.index]
    
    ax.set_xticks(range(len(present_categories)))
    ax.set_xticklabels(present_categories)
    return fig

# Fungsi untuk membuat grafik pie (Pie Chart) berdasarkan jawaban
def plot_pie_chart(df, answered_count, user_answers):
    # Menghitung jumlah jawaban per kategori
    category_counts = pd.Series(user_answers).value_counts()

    # Membuat figure dan axes untuk grafik
    fig, ax = plt.subplots()
    # Menggambar grafik pie
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax, colors=['#6baed6', '#fd8d3c', '#74c476'])
    ax.set_title(f"Distribusi Jawaban ({answered_count} Pertanyaan Dijawab)")
    ax.set_ylabel("")  # Menghilangkan label Y
    return fig

# Fungsi utama untuk aplikasi Streamlit
def main():
    # Memuat dataset dari file CSV
    file_path = "DASS21_v1.csv"  # Sesuaikan dengan lokasi file dataset
    df = pd.read_csv(file_path)

    # Sidebar untuk navigasi antar halaman
    st.sidebar.title("📋 **DASS-21 Assessment**")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigasi Utama")
    menu = st.sidebar.selectbox(
        "Pilih Halaman",  # Pilihan menu navigasi
        ["🏠 Home", "📊 Dataset", "🧠 Prediksi"]
    )

    # Sidebar informasi tambahan tentang aplikasi
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tentang")
    st.sidebar.info(
        """
        Aplikasi ini didasarkan pada **DASS-21**, 
        yang digunakan untuk mengukur tingkat **Stress**, **Anxiety**, dan **Depression**.
        """
    )

    # Halaman Home
    if menu == "🏠 Home":
        st.title("🏠 Aplikasi Prediksi Tingkat Anxiety DASS-21")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""  
            ### Selamat Datang!
            Aplikasi ini dirancang untuk menilai tingkat **Stress**, **Anxiety**, dan **Depression** berdasarkan kuesioner **DASS-21**.
            """, unsafe_allow_html=True)
        with col2:
            st.image("anxiety.jpg", caption="DASS-21 Assessment", use_container_width=True)

    # Halaman Dataset
    elif menu == "📊 Dataset":
        st.title("📊 Dataset DASS-21")
        st.markdown("Berikut adalah dataset dari kuesioner **DASS-21**.")
        st.dataframe(df)

    # Halaman Prediksi
    elif menu == "🧠 Prediksi":
        st.title("🧠 Prediksi Tingkat Anxiety dengan DASS-21")
        st.write("Jawab pertanyaan berikut sesuai dengan kondisi yang dirasakan: ")
        
        user_answers = []  # Tempat untuk menyimpan jawaban
        st.write("Penjelasan Jawaban:")
        st.write("0: Tidak Sama Sekali")
        st.write("1: Sedikit")
        st.write("2: Cukup")
        st.write("3: Sangat")

        # Input jawaban untuk setiap pertanyaan
        for index, row in df.iterrows():
            answer = st.radio(
                f"🔢 {row['qno']} | {row['qtext']} 😟",  # Pertanyaan yang akan ditampilkan
                options=[0, 1, 2, 3],  # Pilihan jawaban 0-3
                index=0,  # Pilihan default adalah 0
                horizontal=True  # Menampilkan radio button secara horizontal
            )
            user_answers.append(answer)  # Menyimpan jawaban

        # Menyiapkan tampilan grafik yang kosong
        st.markdown("### 📈 Pilih Jenis Grafik")
        chart_type = st.selectbox(
            "Pilih jenis grafik yang ingin ditampilkan : (Scroll kebawah untuk melihat hasil)",
            ["Bar Chart", "Line Chart", "Pie Chart"]  # Pilihan jenis grafik
        )
        
        # Menampilkan grafik kosong sebelum tombol ditekan
        fig_placeholder = st.empty()  # Tempat untuk menampilkan grafik kosong
        
        # Tombol untuk melihat hasil
        if st.button("💡 Lihat Hasil"):
            # Menghitung skor berdasarkan jawaban
            total_scores = calculate_scores(df, user_answers)
            st.success("Berikut adalah hasil penilaian:")
            for category, score in total_scores.items():
                st.write(f"**{category.capitalize()}**: {score}")  # Menampilkan hasil per kategori

            # Menampilkan kategori hasil berdasarkan skor
            st.write("### Kategori Hasil:")
            thresholds = {"s": 14, "a": 7, "d": 10}  # Batasan skor untuk kategori
            for category, score in total_scores.items():
                # Klasifikasi skor menjadi "Ringan", "Sedang", atau "Berat"
                level = "Ringan" if score <= thresholds[category] else "Sedang" if score <= thresholds[category] * 2 else "Berat"
                st.write(f"- **{category.capitalize()}**: {level}")

            # Menampilkan grafik berdasarkan pilihan
            if chart_type == "Bar Chart":
                fig = plot_bar_chart(df, len(df), user_answers)
            elif chart_type == "Line Chart":
                fig = plot_line_chart(df, len(df), user_answers)
            else:
                fig = plot_pie_chart(df, len(df), user_answers)

            # Menampilkan grafik
            fig_placeholder.pyplot(fig)  # Update grafik setelah klik tombol
        else:
            fig_placeholder.pyplot(plt.figure())  # Menampilkan grafik kosong sebelum tombol diklik

# Menjalankan aplikasi utama
if __name__ == "__main__":
    main()
