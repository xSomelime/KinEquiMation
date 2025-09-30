import bpy
import json
from pathlib import Path
from collections import deque

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
        # Endast DEF-bones (root exkluderas)
        if not bone.name.startswith("DEF"):
            continue

        parent_name = bone.parent.name if (bone.parent and bone.parent.name.startswith("DEF")) else None

        bones_data[bone.name] = {
            "parent": parent_name,
            "head": [float(v) for v in bone.head],
            "tail": [float(v) for v in bone.tail],
            "roll": float(bone.roll),
            "matrix_local": [[float(x) for x in row] for row in bone.matrix],
            "constraints": []
        }

        if parent_name:
            edges.append({"parent": parent_name, "child": bone.name})

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

    # Paths
    project_root = Path(bpy.path.abspath("//.."))
    out_dir = project_root / "outputs" / "animation_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "skeleton_edges_from_rig.json"
    txt_out = out_dir / "bone_order.txt"

    # Bygg bone_order med BFS (top-down traversal över edges)
    children_map = {}
    roots = []
    for bone, data in bones_data.items():
        parent = data["parent"]
        if parent:
            children_map.setdefault(parent, []).append(bone)
        else:
            roots.append(bone)

    order = []
    queue = deque(roots)
    while queue:
        b = queue.popleft()
        order.append(b)
        for child in children_map.get(b, []):
            queue.append(child)

    # --- Skriv ut JSON med bones + edges + order ---
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"bones": bones_data, "edges": edges, "order": order}, f, indent=2)

    print(f"[INFO] Exporterade {len(bones_data)} DEF-bones och {len(edges)} edges till {json_out}")

    # --- Skriv bone_order.txt med samma ordning ---
    with open(txt_out, "w", encoding="utf-8") as f:
        for name in order:
            f.write(name + "\n")

    print(f"[INFO] Skrev bone_order.txt med {len(order)} DEF-bones till {txt_out}")


if __name__ == "__main__":
    export_skeleton_with_data()
