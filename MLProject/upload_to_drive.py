import os
import sys
import json
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

if len(sys.argv) < 3:
    print("Usage: python upload_to_drive.py <local_folder> <drive_folder_id>")
    sys.exit(1)

local_path = sys.argv[1]
parent_folder_id = sys.argv[2]

if not os.path.exists(local_path):
    print(f"ERROR: Local path not found: {local_path}")
    sys.exit(1)

# Authenticate service account
gauth = GoogleAuth()
gauth.service_account_json = "service_account.json"

with open("service_account.json", "w") as f:
    f.write(os.environ["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"])

gauth.ServiceAuth()
drive = GoogleDrive(gauth)

def upload_file(local_file, parent_id):
    file_name = os.path.basename(local_file)
    f = drive.CreateFile({"title": file_name, "parents": [{"id": parent_id}]})
    f.SetContentFile(local_file)
    f.Upload()
    print(f"Uploaded file: {local_file}")
    return f["id"]

def upload_folder(local_folder, parent_id):
    folder_name = os.path.basename(local_folder)
    folder = drive.CreateFile({"title": folder_name, "mimeType": "application/vnd.google-apps.folder",
                               "parents": [{"id": parent_id}]})
    folder.Upload()
    folder_id = folder["id"]
    print(f"Created folder: {folder_name}, id={folder_id}")

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            upload_file(os.path.join(root, file), folder_id)

upload_folder(local_path, parent_folder_id)
print("Upload completed.")
