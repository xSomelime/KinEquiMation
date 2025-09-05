import csv
import json
from collections import Counter
import os

CSV_PATH = "data/exports/dataset.csv"
JSON_PATH = "data/exports/dataset.json"

# ---- CSV ----
print("=== Kontrollerar CSV ===")
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))

print(f"Antal rader i CSV: {len(reader)}")

# actions i CSV
csv_actions = Counter(r["action"] for r in reader)
print("Actions i CSV:", dict(csv_actions))

# kolla att viktiga fält finns
missing_csv = [r for r in reader if not r["u"] or not r["v"] or not r["head_x"]]
if missing_csv:
    print(f"⚠️ {len(missing_csv)} rader i CSV saknar någon viktig koordinat")
else:
    print("✔ Alla CSV-rader har koordinater")


# ---- JSON ----
print("\n=== Kontrollerar JSON ===")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])
print(f"Antal bilder i JSON: {len(images)}")
print(f"Antal annotationer i JSON: {len(annotations)}")

# actions i JSON
json_actions = Counter(img.get("action") for img in images)
print("Actions i JSON:", dict(json_actions))

# kontrollera att alla annoteringar har extra-data
bad_anns = [a for a in annotations if "extra" not in a or "bone" not in a["extra"]]
if bad_anns:
    print(f"⚠️ {len(bad_anns)} annotationer saknar bone/extra-data")
else:
    print("✔ Alla annotationer har bone/extra-data")

# kolla bildstorlek
bad_imgs = [img for img in images if not img.get("width") or not img.get("height")]
if bad_imgs:
    print(f"⚠️ {len(bad_imgs)} bilder saknar width/height")
else:
    print("✔ Alla bilder har width/height")

# ---- Jämför CSV vs JSON ----
print("\n=== Jämförelse CSV vs JSON ===")
print(f"CSV rader: {len(reader)} | JSON annotationer: {len(annotations)}")
if abs(len(reader) - len(annotations)) < 10:  # liten skillnad kan vara okej
    print("✔ Antal rader stämmer ungefär överens")
else:
    print("⚠️ Stora skillnader mellan CSV och JSON!")

print("\n✔ Kontroll klar")
