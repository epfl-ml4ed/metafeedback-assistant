import json
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def save_to_log(filename, data):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
