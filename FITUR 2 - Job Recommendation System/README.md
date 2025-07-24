# Job Recommender System (Tech Jobs)

Sistem rekomendasi pekerjaan berbasis konten untuk lowongan kerja di bidang teknologi. Proyek ini menggunakan SBERT untuk membuat representasi vektor deskripsi pekerjaan, lalu menghitung kesamaan antar deskripsi untuk memberikan rekomendasi.

---

## 📁 File Penting

Pastikan Anda memiliki file berikut sebelum menjalankan proyek:

- `jobs_data.csv` — Dataset utama berisi deskripsi pekerjaan
- `courses_data.csv` — Dataset utama berisi deskripsi course
- `model_training_jobs.py` — Script untuk melakukan preprocessing, training, dan menyimpan model
- `app.py` — Script utama untuk menjalankan API Flask
- `tester.py` — Script opsional untuk menguji API lokal secara otomatis

---

## 🚀 Langkah Menjalankan Proyek

### 1. Siapkan Dataset

Pastikan file `dealls_jobs_tech.csv` berada di direktori proyek.

### 2. Jalankan Training & Simpan Model

```bash
python model_training_jobs.py
```

Script ini akan:

- Melakukan preprocessing teks
- Membuat embeddings menggunakan SBERT
- Menyimpan model dan embeddings ke dalam folder recommender_models
  Folder yang akan otomatis dibuat:
  recommender_models/
  ├── sbert_embeddings.pkl
  ├── df_for_recommendation.pkl
  └── sbert_model_dir/
  └── ... (isi model SBERT)
- Output file JSON

### 3. Jalankan API secara lokal

```bash
python app.py
```

API akan berjalan di `http://127.0.0.1:5000`

### 4. Uji API

**OPSI 1: Gunakan Postman**

- Method: `POST`
- Endpoint : `http://127.0.0.1:5000/recommend_job_by_course`
- BODY (JSON) :
  {
  "job_description": "Deskripsi pekerjaan yang ingin dicari"
  }

**OPSI 2: Jalankan tester.py**

```bash
python tester.py
```

Script ini akan mengirim request ke API dan mencetak hasil rekomendasinya.

---

## 📌 Catatan

- Pastikan file dan folder sesuai path yang digunakan di script.
- Model SBERT akan otomatis diunduh saat pertama kali dijalankan (pastikan koneksi internet aktif).
