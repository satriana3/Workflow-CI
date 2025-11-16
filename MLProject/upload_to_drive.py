from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import sys

artifact_path = sys.argv[1]

# Ambil JSON dari environment variable
service_account_info = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
if not service_account_info:
    raise Exception("Environment variable GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON tidak ditemukan")

# Tulis sementara ke file
with open("/tmp/service_account.json", "w") as f:
    f.write(service_account_info)

gauth = GoogleAuth()
gauth.ServiceAuthSettings = {
    "client_config_file": "/tmp/service_account.json"
}
gauth.ServiceAuth()
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
