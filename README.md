# Analisis Data Proyek Teralit - Skin Disease Dataset

ID Tim Capstone Project : CC26-PSU247
Judul Proyek : Teralit - Sistem Pendeteksi Penyakit Kulit
Tema yang Dipilih : Healthy Lives & Well-being

Anggota Data Scientist :
- CDCC014D6Y1395 - Putu Krisna Udayana - Data Scientist - [Aktif]
- CDCC014D6Y1872 - I Gede Liyang Anugrah Oktapian - Data Scientist - [Aktif]

Deskripsi singkat: proyek ini berisi analisis dan visualisasi dataset penyakit kulit (format COCO) beserta notebook dan skrip dashboard.

Konten utama
- Dataset: folder `Skin_desease_(Perbaikan)_dataset.coco/` dan `train/` berisi file `_annotations.coco.json` dan gambar.
- Notebook analisis utama: [analisis_data.ipynb](analisis_data.ipynb#L1)
- Skrip dashboard: [dashboard/dashboard.py](dashboard/dashboard.py#L1)
- Output & metadata: folder `output/` (hasil pembersihan, distribusi kelas, dan file anotasi yang sudah dibersihkan).

Persyaratan
- Python 3.8+ (direkomendasikan)
- Install dependensi:

```bash
pip install -r requirements.txt
```

Cara menjalankan
1. Siapkan virtualenv (opsional):

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Buka dan jalankan notebook:

```bash
jupyter notebook analisis_data.ipynb
```

3. Menjalankan dashboard (jika diperlukan):

```bash
python dashboard/dashboard.py
```

Dashboard live (Streamlit)

- Dashboard: https://app-teralit-fmn6grajjkiy2gntdy6sjt.streamlit.app/

Struktur repositori (ringkas)

- analisis_data.ipynb
- requirements.txt
- dashboard/
  - dashboard.py
- output/
  - annotations_cleaned.csv
  - class_distribution.csv
  - images_cleaned.csv
- Skin_desease_(Perbaikan)_dataset.coco/
  - train/
    - _annotations.coco.json
    - *.avif (gambar)

Catatan
- File anotasi besar: pastikan ada cukup ruang disk sebelum menjalankan preprocessing.
- Jika ingin mengonversi gambar `.avif`, gunakan alat konversi sebelum pemrosesan jika library imaging Anda tidak mendukung AVIF.

Lisensi & Kontak
- Lisensi: tentukan lisensi proyek sesuai kebutuhan (mis. MIT).
- Kontak: tambahkan email atau nama pemilik proyek jika ingin info lebih lanjut.
