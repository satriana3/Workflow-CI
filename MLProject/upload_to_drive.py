# MLProject/upload_to_drive.py
import os
import sys
import json
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 2:
    fatal("Usage: upload_to_drive.py <artifact_path>")

artifact_path = sys.argv[1]

# read service account JSON from env
service_account_json = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
if not service_account_json:
    fatal("Environment variable GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON not found. Add it to GitHub Secrets.")

# optional destination folder id
dest_folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")  # optional

# write service account json to temp file
service_json_path = "/tmp/service_account.json"
with open(service_json_path, "w") as f:
    f.write(service_account_json)

# write settings.yaml for PyDrive2
settings_yaml = f"""
client_config_backend: service
service_config:
  client_json_file_path: "{service_json_path}"
"""
settings_path = "/tmp/settings.yaml"
with open(settings_path, "w") as f:
    f.write(settings_yaml)

# verify artifact path existence and normalize
artifact_path = os.path.normpath(artifact_path)
print(f"DEBUG: artifact_path resolved to: {artifact_path}")

if not os.path.exists(artifact_path):
    fatal(f"artifact path not found: {artifact_path}. Make sure MLflow run created artifacts and RUN_ID was set correctly.")

# Authenticate
gauth = GoogleAuth(settings_path)
try:
    gauth.ServiceAuth()
except Exception as e:
    fatal(f"ServiceAuth failed: {e}")

drive = GoogleDrive(gauth)

def upload_file(local_path, parent_id=None):
    fname = os.path.basename(local_path)
    metadata = {'title': fname}
    if parent_id:
        metadata['parents'] = [{'id': parent_id}]
    f = drive.CreateFile(metadata)
    f.SetContentFile(local_path)
    f.Upload()
    print(f"Uploaded file: {local_path} -> {fname} (parent={parent_id})")

def upload_dir(local_dir, parent_id=None):
    # create folder on drive
    folder_title = os.path.basename(local_dir.rstrip("/"))
    folder_metadata = {'title': folder_title, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        folder_metadata['parents'] = [{'id': parent_id}]
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    folder_id = folder['id']
    print(f"Created folder on Drive: {folder_title} (id={folder_id})")
    # upload contents
    for entry in sorted(os.listdir(local_dir)):
        entry_path = os.path.join(local_dir, entry)
        if os.path.isdir(entry_path):
            upload_dir(entry_path, parent_id=folder_id)
        else:
            upload_file(entry_path, parent_id=folder_id)

# If a destination folder id is provided, ensure it exists (we assume it's valid)
parent_id = dest_folder_id if dest_folder_id else None

if os.path.isdir(artifact_path):
    # upload directory recursively into either dest folder or create folder from artifact name
    if parent_id:
        # create a subfolder with run artifact base name inside parent
        subfolder_title = os.path.basename(artifact_path.rstrip("/"))
        subfolder_metadata = {'title': subfolder_title, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': parent_id}]}
        subfolder = drive.CreateFile(subfolder_metadata)
        subfolder.Upload()
        subfolder_id = subfolder['id']
        print(f"Created subfolder in Drive parent {parent_id}: {subfolder_title} (id={subfolder_id})")
        for entry in sorted(os.listdir(artifact_path)):
            entry_path = os.path.join(artifact_path, entry)
            if os.path.isdir(entry_path):
                upload_dir(entry_path, parent_id=subfolder_id)
            else:
                upload_file(entry_path, parent_id=subfolder_id)
    else:
        # create folder in root
        upload_dir(artifact_path, parent_id=None)
else:
    # single file
    upload_file(artifact_path, parent_id=parent_id)

print("Upload completed successfully.")
