import os
import json
import sys
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ======== INPUT FROM WORKFLOW ===========
artifact_path = sys.argv[1]

# ======== LOAD SERVICE ACCOUNT JSON FROM SECRET ==========
service_account_json = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")

if not service_account_json:
    raise Exception("Environment variable GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON not found")

# Write JSON ke file sementara
service_json_path = "/tmp/service_account.json"
with open(service_json_path, "w") as f:
    f.write(service_account_json)

# ======== WRITE Pydrive2 SETTINGS FILE ==========
settings_yaml = """
client_config_backend: service
service_config:
  client_json_file_path: "/tmp/service_account.json"
"""

settings_path = "/tmp/settings.yaml"
with open(settings_path, "w") as f:
    f.write(settings_yaml)

# ======== AUTHENTICATE USING SERVICE ACCOUNT ==========
gauth = GoogleAuth(settings_path)
gauth.ServiceAuth()

drive = GoogleDrive(gauth)


# ======== UPLOAD FILE OR FOLDER ==========
def upload_file(path):
    file_drive = drive.CreateFile({
        "title": os.path.basename(path)
    })
    file_drive.SetContentFile(path)
    file_drive.Upload()
    print(f"Uploaded: {path}")


if os.path.isdir(artifact_path):
    for fname in os.listdir(artifact_path):
        file_path = os.path.join(artifact_path, fname)
        upload_file(file_path)
else:
    upload_file(artifact_path)

print("Upload completed successfully.")
