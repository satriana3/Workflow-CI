from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import json
import sys
import os

artifact_path = sys.argv[1]
service_account_json = sys.argv[2]

# simpan JSON ke file sementara
with open("/tmp/service_account.json", "w") as f:
    f.write(service_account_json)

gauth = GoogleAuth()
gauth.credentials = None
gauth.ServiceAuth(settings_file="/tmp/service_account.json")
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
