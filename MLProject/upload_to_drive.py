import os
import sys
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 3:
    fatal("Usage: python upload_to_drive.py <local_folder> <drive_folder_id>")

local_root = os.path.normpath(sys.argv[1])
parent_folder_id = sys.argv[2]

if not os.path.exists(local_root):
    fatal(f"Local path not found: {local_root}")

# Load service account JSON from secret
sa_json = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
if not sa_json:
    fatal("Missing GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON secret")

# Write service account JSON to temporary file
service_json_path = "/tmp/service_account.json"
with open(service_json_path, "w") as f:
    f.write(sa_json)

# Authenticate using service account
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(
    service_json_path, scopes=SCOPES
)
service = build('drive', 'v3', credentials=credentials)

# Helper function to create folder in Google Drive
def create_folder(name, parent_id):
    metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=metadata, fields='id').execute()
    return folder['id']

# Helper function to upload a single file
def upload_file(file_path, parent_id):
    file_name = os.path.basename(file_path)
    media = MediaFileUpload(file_path, resumable=True)
    file_metadata = {'name': file_name, 'parents': [parent_id]}
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {file_path} -> {file_name}")

# Recursive upload function
def upload_dir(local_dir, parent_id):
    folder_name = os.path.basename(local_dir.rstrip(os.sep))
    folder_id = create_folder(folder_name, parent_id)
    for entry in sorted(os.listdir(local_dir)):
        path = os.path.join(local_dir, entry)
        if os.path.isdir(path):
            upload_dir(path, folder_id)
        else:
            upload_file(path, folder_id)

# Start upload
upload_dir(local_root, parent_folder_id)
print("Upload finished successfully.")
