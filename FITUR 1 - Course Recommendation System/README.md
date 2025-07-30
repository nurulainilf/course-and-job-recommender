# 🎓 Course Recommender System

Sistem rekomendasi kursus berbasis *Content-Based Filtering* yang dapat memberikan rekomendasi kursus untuk menutup gap skill pengguna dan disesuaikan dengan industri pekerjaan impian pengguna.

## 📁 Struktur Folder
```text
├── app.py                         # Endpoint Flask utama
├── recommender.py                 # Fungsi pemrosesan rekomendasi
├── test.py                        # Script untuk mengetes API secara lokal
├── course_scraping.ipynb          # Notebook untuk scraping data kursus
├── course_preprocessing.ipynb     # Notebook untuk preprocessing dan embedding
├── PP_MLOps_Nurul Ainil Fitri_Pricillia Silfany_Output.json      # Output hasil rekomendasi
├── requirements.txt               # Daftar dependencies
│
├── data/
│   ├── courses_data_raw.csv       # Data mentah hasil scraping
│   ├── courses_data.json          # Data kursus yang sudah dibersihkan
│   └── course_embeddings.pkl      # Embedding vektorisasi course
│
├── models/
│   └── sentence_model/            # Folder model SBERT yang digunakan
```

## 📦 Instalasi Dependencies
Ikuti langkah berikut untuk menginstal semua library yang dibutuhkan:
1. **Aktifkan virtual environment** (jika sudah dibuat) dengan menulis perintah berikut diterminal:
   ```
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # MacOS/Linux
   ```
2. **Instal library** yang dibutuhkan dengan menjalankan perintah berikut di terminal:
   ```
   pip install -r requirements.txt
   ```

## ⚙ Cara Menjalankan Proyek

### 1️⃣ Preprocessing (Opsional)
> Lewati langkah ini *jika file berikut sudah tersedia*:  
> - `data/courses_data.json`  
> - `data/course_embeddings.pkl`  
> - `models/sentence_model/`

1. Pastikan file `courses_data_raw.csv` tersedia di folder data/.
2. Jalankan notebook `course_preprocessing.ipynb` dan klik **Run All** untuk:
   - Membersihkan dan memformat data kursus
   - Melakukan tokenisasi dan vektorisasi dengan **SentenceTransformer**
   - Menyimpan hasil sebagai:
     - `data/courses_data.json` (versi bersih dalam format JSON)
     - `data/course_embeddings.pkl` (embedding hasil vektorisasi)
     - `models/sentence_model/` (folder model SBERT yang digunakan)

---

### 2️⃣ Menjalankan Sistem Rekomendasi

#### Cara 1: Jalankan Secara Lokal

1. Buka terminal dan pastikan sudah berada pada direktori utama proyek
2. Jalankan file app.py dengan click "Run" atau tulis di terminal dengan perintah:
    ```
    python app.py
    ```    
3. Server akan berjalan di:
    ```
    http://localhost:5000
    ```
4. Jalankan file test.py untuk mengirim permintaan ke endpoint dengan cara menulis di terminal:
    ```
    python test.py
    ```
5. Output akan tersimpan dengan nama:
    ```
    PP_MLOps_Nurul Ainil Fitri_Pricillia Silfany_Output.json
    ```

#### Cara 2: Menggunakan Postman

1. **Buka Postman** *(unduh dari [postman.com/downloads](https://www.postman.com/downloads) jika belum terinstal)*
2. **Jalankan Flask server**  
   Jalankan perintah di terminal:
    ```
    python app.py
    ```
   Maka server akan aktif di: `http://localhost:5000`

4. **Buat request baru di Postman**:
- Pilih metode: `POST`
- URL: `http://localhost:5000/recommend`
- Klik tab **"Body"** → pilih **"raw"** → ubah format ke **JSON**
- Masukkan payload seperti contoh berikut (bisa diubah dan disesuaikan):

 ```json
 {
   "skills": ["data analysis", "python", "sql"],
   "category": "Data Science"
 }
 ```

4. **Klik tombol "Send"**

5. **Lihat hasil rekomendasi di bagian "Response"** 
Jika berhasil, kamu akan mendapatkan daftar rekomendasi kursus dalam format JSON, misalnya:

 ```json
 {
   "recommendations": [
     {
       "title": "Data Science with Python",
       "provider": "Coursera",
       "similarity_score": 0.87
     },
     ...
   ]
 }
 ```

6. **(Opsional) Simpan response ke file** 
Klik ikon titik tiga di kanan atas hasil response → `Save Response`

## 📊 Evaluasi Performa Sistem Rekomendasi

Evaluasi dilakukan menggunakan metrik **Precision@10**, yaitu rasio jumlah kursus relevan (berdasarkan label *ground truth*) yang termasuk dalam 10 rekomendasi teratas. Evaluasi mencakup 7 test case yang merepresentasikan berbagai bidang dalam industri teknologi, berdasarkan penilaian dari 7 pengguna. Berikut hasil evaluasinya:

| Test Case Field         | Precision@10 |
|-------------------------|--------------|
| Project Management      | 0.4          |
| Front-end Development   | 0.7          |
| Cybersecurity           | 0.8          |
| Digital Marketing       | 0.9          |
| Data Science            | 0.7          |
| UI/UX Design            | 0.8          |
| Game Development        | 0.7          |

📈 **Rata-rata Precision@10: 0.7143**

Hasil ini menunjukkan bahwa sistem mampu memberikan rekomendasi yang cukup relevan dengan kebutuhan pengguna, dengan rata-rata 7 dari 10 kursus yang direkomendasikan sesuai dengan ground truth yang telah ditentukan.
