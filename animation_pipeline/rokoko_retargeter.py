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

    # 1. Låt Rokoko bygga sin lista först
    bpy.ops.rsl.build_bone_list()
    bpy.context.view_layer.update()
    print("[INFO] Rokoko bone list byggd")

    # 2. Patch-funktion som körs efter en delay
    def patch_mapping():
        updated, skipped = 0, 0
        bone_list = bpy.context.scene.rsl_retargeting_bone_list

        if not bone_list:
            print("[WARN] Rokoko bone list är tom, försöker igen senare...")
            return 0.5  # prova igen om 0.5 sek

        for item in bone_list:
            # Rokoko använder "custom_bone_DEF-..." som source
            def_bone = item.bone_name_source.replace("custom_bone_", "")
            if def_bone in mapping_data:
                target_bone = mapping_data[def_bone]["controller"]
                if target_bone:
                    old_target = item.bone_name_target
                    item.bone_name_target = target_bone
                    item.is_custom = True
                    updated += 1
                    print(f"[REPLACE] {def_bone} : {old_target} → {target_bone}")
            else:
                skipped += 1

        print(f"[INFO] Uppdaterade {updated} mappings, hoppade över {skipped}.")

        # Debug: skriv slutliga targets
        for i, item in enumerate(bone_list):
            print(f"{i:03d} | source={item.bone_name_source}, target={item.bone_name_target}, is_custom={item.is_custom}")

        return None  # stoppa timern efter en körning

    # Registrera patch-funktionen att köras efter 0.5 sek
    bpy.app.timers.register(patch_mapping, first_interval=0.5)


if __name__ == "__main__":
    project_root = Path(bpy.path.abspath("//.."))
    json_path = project_root / "outputs" / "animation_pipeline" / "controller_mapping.json"
    load_mapping_into_rokoko(str(json_path))
