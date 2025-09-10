# dataset_pipeline/blender_export/export_skeleton_meta.py
# Exporterar benordning (def_bones.txt) och skelettkanter (skeleton_edges.json)
# från aktiv rigg i Blender, i format som matchar synth_to_coco.py.

import bpy
import json
import os

def main():
    # ---- Ställ in output-mapp ----
    blend_dir = bpy.path.abspath("//")        # mappen där .blend ligger (assets/)
    project_root = os.path.abspath(os.path.join(blend_dir, ".."))  # ett steg upp
    out_dir = os.path.join(project_root, "dataset_pipeline", "data", "dataset_exports")
    os.makedirs(out_dir, exist_ok=True)

    bones_txt_path = os.path.join(out_dir, "def_bones.txt")
    edges_json_path = os.path.join(out_dir, "skeleton_edges.json")

    # ---- Hitta armature ----
    arm = None
    preferred_names = ["rig", "rig.001"]

    for name in preferred_names:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "ARMATURE":
            arm = obj
            break

    # Fallback: ta första armature om inget matchade
    if arm is None:
        armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        if armatures:
            arm = armatures[0]

    if arm is None:
        print("[!] Ingen armature hittad i scenen!")
        return

    print(f"[export_skeleton_meta] Hittade armature: {arm.name}")



    # ---- Samla DEF-bones ----
    pose_bones = [
    pb for pb in arm.pose.bones
    if pb.name.startswith("DEF-") and not pb.name.startswith(("DEF-mane_base", "DEF-mane_top"))
    ]

    if not pose_bones:
        print("[!] Inga DEF-* ben hittades!")
        return

    # Ordna alfabetiskt för konsekvent ordning
    bones_sorted = sorted([pb.name for pb in pose_bones])

    # Skriv def_bones.txt
    with open(bones_txt_path, "w", encoding="utf-8") as f:
        for name in bones_sorted:
            f.write(name + "\n")
    print(f"[export_skeleton_meta] Skrev {len(bones_sorted)} ben till {bones_txt_path}")

    # ---- Skapa kanter (parent → child) ----
    name_set = set(bones_sorted)
    edges = []
    for pb in pose_bones:
        if pb.parent and pb.parent.name in name_set:
            parent_name = pb.parent.name
            edges.append([parent_name, pb.name])

    # Spara skeleton_edges.json i rätt format (med både bones & edges)
    skel = {"bones": bones_sorted, "edges": edges}
    with open(edges_json_path, "w", encoding="utf-8") as f:
        json.dump(skel, f, ensure_ascii=False, indent=2)
    print(f"[export_skeleton_meta] Skrev {len(edges)} kanter till {edges_json_path}")
    print(f"[OK] Export klar. Filer sparade i: {out_dir}")

if __name__ == "__main__":
    main()
