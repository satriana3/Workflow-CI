# MLProject/upload_to_drive.py
import os
import sys
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 2:
    fatal("Usage: upload_to_drive.py <artifact_path>")

artifact_path = sys.argv[1]
artifact_path = os.path.normpath(artifact_path)
print(f"DEBUG: artifact_path -> {artifact_path}")

service_account_json = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
if not service_account_json:
    fatal("Missing env var GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON (set it as GitHub Secret)")

dest_folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")  # optional

# write service account json to temp file
service_json_path = "/tmp/service_account.json"
with open(service_json_path, "w") as f:
    f.write(service_account_json)

# write pydrive2 settings yaml
settings_yaml = f"""
client_config_backend: service
service_config:
  client_json_file_path: "{service_json_path}"
"""
settings_path = "/tmp/settings.yaml"
with open(settings_path, "w") as f:
    f.write(settings_yaml)

# verify artifact exists
if not os.path.exists(artifact_path):
    fatal(f"Artifact path not found: {artifact_path}")

# authenticate
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

def create_folder_and_get_id(folder_name, parent_id=None):
    md = {'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        md['parents'] = [{'id': parent_id}]
    folder = drive.CreateFile(md)
    folder.Upload()
    return folder['id']

def upload_dir(local_dir, parent_id=None):
    base = os.path.basename(local_dir.rstrip(os.sep))
    if parent_id:
        folder_id = create_folder_and_get_id(base, parent_id=parent_id)
    else:
        folder_id = create_folder_and_get_id(base)
    print(f"Created folder {base} with id {folder_id}")
    for entry in sorted(os.listdir(local_dir)):
        ep = os.path.join(local_dir, entry)
        if os.path.isdir(ep):
            upload_dir(ep, parent_id=folder_id)
        else:
            upload_file(ep, parent_id=folder_id)

# perform upload
if os.path.isdir(artifact_path):
    if dest_folder_id:
        upload_dir(artifact_path, parent_id=dest_folder_id)
    else:
        upload_dir(artifact_path, parent_id=None)
else:
    upload_file(artifact_path, parent_id=dest_folder_id if dest_folder_id else None)

print("Upload finished.")
