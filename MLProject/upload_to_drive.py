import os
import sys
import json
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 3:
    fatal("Usage: python upload_to_drive.py <local_folder> <drive_folder_id>")

local_root = os.path.normpath(sys.argv[1])
parent_folder_id = sys.argv[2]

if not os.path.exists(local_root):
    fatal(f"Local path not found: {local_root}")

# Tulis service account JSON ke file
sa_json = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
if not sa_json:
    fatal("Missing GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON secret")

service_json_path = "/tmp/service_account.json"
with open(service_json_path, "w") as f:
    # pastikan format JSON valid
    f.write(json.dumps(json.loads(sa_json)))

# Authenticate
gauth = GoogleAuth()
gauth.settings['client_config_file'] = service_json_path
gauth.ServiceAuth()  # <--- tidak ada argumen di sini
drive = GoogleDrive(gauth)

# Helper functions
def create_folder(name, parent_id):
    md = {
        "title": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"id": parent_id}]
    }
    f = drive.CreateFile(md)
    f.Upload()
    return f["id"]

def upload_file(path, parent_id):
    fname = os.path.basename(path)
    meta = {"title": fname, "parents": [{"id": parent_id}]}
    f = drive.CreateFile(meta)
    f.SetContentFile(path)
    f.Upload()
    print(f"Uploaded file: {path} -> {fname}")

def upload_dir(local_dir, parent_id):
    base = os.path.basename(local_dir.rstrip(os.sep))
    folder_id = create_folder(base, parent_id)
    for entry in sorted(os.listdir(local_dir)):
        ep = os.path.join(local_dir, entry)
        if os.path.isdir(ep):
            upload_dir(ep, folder_id)
        else:
            upload_file(ep, folder_id)

# Start upload
upload_dir(local_root, parent_folder_id)
print("Upload finished.")
