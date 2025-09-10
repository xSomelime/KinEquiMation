# dataset_pipeline/tools/check_coco.py
import json

path = "dataset_pipeline/data/dataset_exports/coco_synth_68.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("✅ Nycklar i JSON:", list(data.keys()))

# --- Categories ---
cats = data.get("categories", [])
if not cats:
    print("⚠️ Inga categories hittades!")
else:
    cat = cats[0]
    print("\n📂 Category info:")
    print("  name:", cat.get("name"))
    print("  keypoints:", len(cat.get("keypoints", [])))
    print("  skeleton edges:", len(cat.get("skeleton", [])))

# --- Images ---
print(f"\n🖼️ Antal bilder: {len(data.get('images', []))}")
if data.get("images"):
    print("Exempel på första bilden:", data["images"][0])

# --- Annotations ---
anns = data.get("annotations", [])
print(f"\n✏️ Antal annotationer: {len(anns)}")

bad_keypoints = []
bad_bbox = []

for ann in anns:
    kpts = ann.get("keypoints", [])
    if len(kpts) != 68 * 3:
        bad_keypoints.append((ann["id"], len(kpts)))
    if "bbox" not in ann or len(ann["bbox"]) != 4:
        bad_bbox.append(ann["id"])

print(f"\n🔎 Annotation-kontroll:")
if not bad_keypoints:
    print("  ✔ Alla annotationer har 204 värden i 'keypoints'")
else:
    print(f"  ❌ {len(bad_keypoints)} annotationer har fel antal keypoints! Exempel:", bad_keypoints[:5])

if not bad_bbox:
    print("  ✔ Alla annotationer har bbox[4]")
else:
    print(f"  ❌ {len(bad_bbox)} annotationer saknar bbox eller har fel format. Exempel:", bad_bbox[:5])

# --- num_keypoints check ---
wrong_numkp = []
for ann in anns:
    kpts = ann.get("keypoints", [])
    numkp = ann.get("num_keypoints", -1)
    vis_count = sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0)
    if vis_count != numkp:
        wrong_numkp.append((ann["id"], numkp, vis_count))

if not wrong_numkp:
    print("  ✔ Alla 'num_keypoints' matchar faktisk synliga keypoints")
else:
    print(f"  ❌ {len(wrong_numkp)} annotationer har mismatch mellan 'num_keypoints' och faktiska synliga. Exempel:", wrong_numkp[:5])
