# Exporterar DEF-ben till CTRL-ben mappningar


import bpy
import json
from pathlib import Path

def export_bone_mappings():
    rig = bpy.data.objects.get("rig")
    if not rig or rig.type != "ARMATURE":
        raise RuntimeError("Hittar ingen armature som heter 'rig'")

    data = {}
    for bone in rig.data.bones:
        if not bone.name.startswith("DEF"):
            continue

        entry = {
            "parent": bone.parent.name if bone.parent else None,
            "head": list(bone.head_local),
            "tail": list(bone.tail_local),
            "constraints": []
        }

        if bone.name in rig.pose.bones:
            pb = rig.pose.bones[bone.name]
            for c in pb.constraints:
                entry["constraints"].append({
                    "name": c.name,
                    "type": c.type,
                    "target": getattr(c.target, "name", None)
                })

        data[bone.name] = entry

    project_root = Path(bpy.path.abspath("//.."))
    out_dir = project_root / "outputs" / "animation_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "def_to_ctrl_map.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"bones": data}, f, indent=2)

    print(f"[INFO] Exporterade {len(data)} DEF-bones till {out_path}")

export_bone_mappings()
