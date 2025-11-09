# Workflow-CI
Membuat workflow CI menggunakan MLflow Project agar dapat melakukan re-training model secara otomatis ketika trigger dipantik. 

# Student Performance Prediction — MLflow CI/CD Pipeline

Pipeline ini melatih model untuk memprediksi **tingkat performa siswa** berdasarkan data nilai dan faktor-faktor demografis, serta secara otomatis membuat dan mempublikasikan **Docker Image** ke Docker Hub.

## 🚀 Fitur Utama

- ✅ **Automated ML Pipeline (CI/CD)** menggunakan GitHub Actions  
- ✅ **MLflow Tracking** untuk pencatatan eksperimen dan artefak model  
- ✅ **Build dan Push Docker Image** ke Docker Hub  
- ✅ **Upload hasil training (model & metrics)** ke GitHub Artifacts  
- ✅ Dapat dijalankan ulang otomatis setiap ada perubahan pada branch `main`

## 🧩 Struktur Folder

Workflow-CI
├── .github/
│ └── workflows/
│ └── workflow-ci.yml # File workflow GitHub Actions
├── MLProject/
| └── modelling.py
| └── conda.yaml
| └── MLProject
| ├── studentsperformance_preprocessing/
| |  └── StudentsPerformance_preprocessing.csv
| └── Tautan ke Docker Hub
└── README.md 


## ⚙️ Tahapan Workflow CI

Pipeline otomatis dijalankan setiap kali ada perubahan pada branch `main`.  
Berikut tahapan yang dilakukan secara otomatis oleh **GitHub Actions**:

1. **Checkout Repository**
   - Mengambil kode terbaru dari branch `main`.

2. **Setup Python**
   - Menggunakan versi **Python 3.12.7** sesuai environment `conda.yaml`.

3. **Install Dependencies**
   - Menginstal library seperti `mlflow`, `scikit-learn`, `pandas`, `numpy`, dll.

4. **Run MLflow Project**
   - Menjalankan `modelling.py` untuk melatih model.
   - Semua eksperimen dan model tersimpan otomatis di folder `mlruns`.

5. **Show Latest Run Info**
   - Menampilkan informasi `RUN_ID` dan lokasi artefak model.

6. **Upload Artifacts to GitHub**
   - Mengunggah model dan hasil training ke GitHub sebagai artifacts.

7. **Login to Docker Hub**
   - Autentikasi ke akun Docker Hub menggunakan `secrets.DOCKERHUB_TOKEN`.

8. **Build and Push Docker Image**
   - Membangun image MLflow model dan mengunggah ke Docker Hub:
     ```
     docker push satriana3/student-performance-mlflow:latest
     ```

## 🧠 Model Machine Learning

Model yang digunakan: **Random Forest Regressor / Classifier**

### Input:
Dataset: `StudentsPerformance_preprocessing.csv`

### Output:
Model MLflow yang dapat disimpan dan digunakan kembali (`MLmodel` format).

---

## 🧪 Cara Menjalankan Secara Manual

Project ini bisa juga dijalankan di lokal dengan perintah:

cd MLProject
mlflow run . -P data_path=studentsperformance_preprocessing/StudentsPerformance_preprocessing.csv --env-manager=local

Docker Image

Setelah pipeline berjalan sukses, Docker image akan otomatis terunggah ke:

👉 Docker Hub - satriana3/student-performance-mlflow

Untuk menjalankan container-nya:
docker pull satriana3/student-performance-mlflow:latest
docker run -p 5000:8080 satriana3/student-performance-mlflow:latest

☁️ Penyimpanan Artefak
Model hasil training dan metrics disimpan sebagai GitHub Actions Artifacts

👩‍💻 Developer
Nama: Satriana
Role: Machine Learning 
Tools: Python • MLflow • GitHub Actions • Docker • Scikit-learn
