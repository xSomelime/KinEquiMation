import bpy
import json
from pathlib import Path

def load_mapping_into_rokoko(json_path, source_armature="ML_rig", target_armature="rig"):
    with open(json_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    if "bones" in mapping_data:
        mapping_data = mapping_data["bones"]

    source_obj = bpy.data.objects.get(source_armature)
    target_obj = bpy.data.objects.get(target_armature)
    if source_obj is None or target_obj is None:
        raise RuntimeError(f"Kunde inte hitta {source_armature} eller {target_armature} i scenen!")

    bpy.context.scene.rsl_retargeting_armature_source = source_obj
    bpy.context.scene.rsl_retargeting_armature_target = target_obj
    print(f"[INFO] Source set to {source_obj.name}, Target set to {target_obj.name}")

    # Bygg Rokoko’s egen bone list först
    bpy.ops.rsl.build_bone_list()

    updated, skipped = 0, 0
    for def_bone, entry in mapping_data.items():
        target_bone = entry.get("controller")
        if not target_bone:
            skipped += 1
            continue

        for item in bpy.context.scene.rsl_retargeting_bone_list:
            if item.bone_name_source == def_bone:
                # Ersätt alltid Rokokos auto-target om det är DEF/MCH/ORG
                if item.bone_name_target.startswith(("DEF", "MCH", "ORG")):
                    item.bone_name_target = target_bone
                    item.is_custom = True
                    updated += 1
                    print(f"[REPLACE] {def_bone} → {target_bone}")
                else:
                    print(f"[KEEP] {def_bone} redan satt till {item.bone_name_target}")
                break

    print(f"[INFO] Uppdaterade {updated} mappings, hoppade över {skipped} (saknar controller).")

    # Debug: skriv slutliga targets
    for i, item in enumerate(bpy.context.scene.rsl_retargeting_bone_list):
        print(f"{i:03d} | source={item.bone_name_source}, target={item.bone_name_target}, is_custom={item.is_custom}")


if __name__ == "__main__":
    project_root = Path(bpy.path.abspath("//.."))
    json_path = project_root / "outputs" / "animation_pipeline" / "def_to_ctrl_map.json"
    load_mapping_into_rokoko(str(json_path))
