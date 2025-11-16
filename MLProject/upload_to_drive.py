from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import sys

artifact_path = sys.argv[1]

# Auth otomatis via command line (langsung jalan di GitHub Actions)
gauth = GoogleAuth()
gauth.CommandLineAuth()  # Non-interactive OAuth
drive = GoogleDrive(gauth)

def upload_file(file_path):
    f = drive.CreateFile({'title': os.path.basename(file_path)})
    f.SetContentFile(file_path)
    f.Upload()
    print(f"Uploaded {file_path} to Google Drive successfully.")

if os.path.isdir(artifact_path):
    for fname in os.listdir(artifact_path):
        upload_file(os.path.join(artifact_path, fname))
else:
    upload_file(artifact_path)
