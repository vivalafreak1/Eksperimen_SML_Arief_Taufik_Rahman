# 🚀 Telco Customer Churn - End to End MLOps Pipeline

Repositori ini berisi proyek akhir (_Submission_) untuk kelas **Machine Learning Operations (MLOps)**. Proyek ini mengimplementasikan alur kerja MLOps secara menyeluruh (End-to-End), mulai dari pemrosesan data otomatis, eksperimen model, CI/CD, hingga sistem pemantauan (monitoring) terintegrasi.

**Oleh:** Arief Taufik Rahman

## 🛠️ Tech Stack & Tools

- **Machine Learning:** Scikit-Learn, XGBoost, Pandas
- **Tracking & Registry:** MLflow, DagsHub
- **CI/CD:** GitHub Actions, MLflow Projects
- **Containerization:** Docker & Docker Hub
- **Monitoring & Alerting:** Prometheus, Grafana, Flask

---

## 📂 Struktur Proyek & Pencapaian Kriteria

Proyek ini diselesaikan dengan target pencapaian **Advance** pada seluruh kriteria penilaian:

### 1. Data Preprocessing (CI)

- Menggunakan skrip Python (`automate_Arief_Taufik_Rahman.py`) untuk membersihkan data mentah.
- Diotomatisasi menggunakan **GitHub Actions** (`preprocessing.yml`). Skrip berjalan otomatis setiap kali ada perubahan pada folder `preprocessing/`, dan hasil data bersih (`telco_processed.csv`) di-_push_ kembali ke repositori secara otomatis.

### 2. Membangun Model & Tracking Eksperimen

- Menggunakan algoritma **XGBoost Classifier** dengan teknik optimasi hyperparameter (GridSearchCV).
- Metrik, parameter, dan artefak model (Confusion Matrix, Feature Importance) dicatat secara otomatis menggunakan **MLflow**.
- _Tracking server_ di-host secara _remote_ menggunakan integrasi **DagsHub**.

### 3. Workflow CI/CD & Dockerization

- Diotomatisasi menggunakan **GitHub Actions** (`cicd.yml`) dan standar **MLflow Project** (`MLproject` & `conda.yaml`).
- Setiap kode didorong (push) ke branch `main`, GitHub Actions akan menjalankan proses _training_ di _environment_ terisolasi.
- Model terbaik yang dihasilkan otomatis di-_build_ menjadi **Docker Image** dan dikirim (push) ke **Docker Hub**.

### 4. Sistem Monitoring & Logging

- Model di-_serve_ menggunakan kontainer Docker secara lokal.
- Menggunakan **Flask** sebagai jembatan (_Prometheus Exporter_) untuk menangkap metrik prediksi dan performa sistem.
- **Prometheus** digunakan untuk menarik (_scrape_) data metrik (CPU, RAM, Total Requests, Latency, Throughput).
- **Grafana** digunakan untuk visualisasi _dashboard_ waktu nyata (real-time) dengan 10+ panel metrik.
- Mengonfigurasi **Grafana Alerting** (SMTP) untuk mengirimkan notifikasi _email_ otomatis ketika terjadi anomali (misal: penggunaan CPU > 80% atau lonjakan _request_).

---

## 🚀 Cara Menjalankan Repositori Ini (Lokal)

**1. Clone repositori & Install dependencies**

```bash
git clone [https://github.com/USERNAME_ANDA/Eksperimen_SML_Arief_Taufik_Rahman.git](https://github.com/USERNAME_ANDA/Eksperimen_SML_Arief_Taufik_Rahman.git)
cd Eksperimen_SML_Arief_Taufik_Rahman
python -m venv venv
source venv/Scripts/activate  # Untuk Windows
pip install -r requirements.txt
```

**2. Jalankan Data Preprocessing**

```bash
cd preprocessing
python automate_Arief_Taufik_Rahman.py
```

**3. Jalankan Eksperimen Model (Memerlukan akun DagsHub)**

```bash
cd Membangun_model
# Pastikan sudah login dagshub: dagshub login
python modelling_tuning.py
```

**4. Pull & Run Docker Image (Serving)**

```bash
docker pull eddiesti22/telco-churn:latest
docker run -p 5005:8080 eddiesti22/telco-churn:latest
```
