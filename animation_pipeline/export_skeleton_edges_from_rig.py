# animation_pipeline/export_skeleton_edges_from_rig.py
import bpy
import json
from pathlib import Path

def export_skeleton_with_data():
    rig = bpy.data.objects.get("rig")
    if not rig or rig.type != "ARMATURE":
        raise RuntimeError("Hittar ingen armature som heter 'rig'!")

    # Växla till EDIT mode för att läsa head/tail/roll
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")

    bones_data = {}
    edges = []

    for bone in rig.data.edit_bones:
        if not bone.name.startswith("DEF"):
            continue

        bones_data[bone.name] = {
            "parent": bone.parent.name if (bone.parent and bone.parent.name.startswith("DEF")) else None,
            "head": [float(v) for v in bone.head],
            "tail": [float(v) for v in bone.tail],
            "roll": float(bone.roll),
            "constraints": []
        }

        if bone.parent and bone.parent.name.startswith("DEF"):
            edges.append({"parent": bone.parent.name, "child": bone.name})

    # Tillbaka till OBJECT mode innan constraints
    bpy.ops.object.mode_set(mode="OBJECT")

    # Lägg till constraints (från pose-bones)
    for bone_name, data in bones_data.items():
        pb = rig.pose.bones.get(bone_name)
        if not pb:
            continue
        for c in pb.constraints:
            data["constraints"].append({
                "name": c.name,
                "type": c.type,
                "target": getattr(c.target, "name", None)
            })

    # Skriv ut JSON
    project_root = Path(bpy.path.abspath("//.."))
    out_dir = project_root / "outputs" / "animation_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "skeleton_edges_from_rig.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"bones": bones_data, "edges": edges}, f, indent=2)

    print(f"[INFO] Exporterade {len(bones_data)} DEF-bones (med roll) och {len(edges)} edges till {out_path}")


if __name__ == "__main__":
    export_skeleton_with_data()
