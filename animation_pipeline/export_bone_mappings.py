# Exporterar DEF-ben till slutgiltig controller-mapping

import bpy
import json
from pathlib import Path

def resolve_controller(rig, bone_name, visited=None):
    """Följ constraint-kedjan tills vi hittar sista controllern."""
    if visited is None:
        visited = set()
    if bone_name in visited:
        return None  # undvik loopar
    visited.add(bone_name)

    pb = rig.pose.bones.get(bone_name)
    if not pb:
        return None

    for c in pb.constraints:
        target_name = getattr(c.target, "name", None)
        if not target_name or target_name not in rig.data.bones:
            continue

        # Om target är en DEF/MCH/ORG -> fortsätt följa kedjan
        if target_name.startswith(("DEF", "MCH", "ORG")):
            resolved = resolve_controller(rig, target_name, visited)
            if resolved:
                return resolved
            continue

        # Annars har vi hittat en controller
        return target_name

    return None


def export_bone_mappings():
    rig = bpy.data.objects.get("rig")
    if not rig or rig.type != "ARMATURE":
        raise RuntimeError("Hittar ingen armature som heter 'rig'")

    data = {}
    for bone in rig.data.bones:
        if not bone.name.startswith("DEF"):
            continue

        controller = resolve_controller(rig, bone.name)

        entry = {
            "parent": bone.parent.name if bone.parent else None,
            "head": list(bone.head_local),
            "tail": list(bone.tail_local),
            "controller": controller
        }

        data[bone.name] = entry
        print(f"[MAP] {bone.name} → {controller}")

    project_root = Path(bpy.path.abspath("//.."))
    out_dir = project_root / "outputs" / "animation_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "def_to_ctrl_map.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"bones": data}, f, indent=2)

    print(f"[INFO] Exporterade {len(data)} DEF-bones till {out_path}")

export_bone_mappings()
# Exporterar DEF-ben till slutgiltig controller-mapping

import bpy
import json
from pathlib import Path

def resolve_controller(rig, bone_name, visited=None):
    """Följ constraint-kedjan tills vi hittar sista controllern."""
    if visited is None:
        visited = set()
    if bone_name in visited:
        return None  # undvik loopar
    visited.add(bone_name)

    pb = rig.pose.bones.get(bone_name)
    if not pb:
        return None

    for c in pb.constraints:
        target_name = getattr(c, "subtarget", None)
        if not target_name or target_name not in rig.data.bones:
            continue

        # Om target är en DEF/MCH/ORG -> fortsätt följa kedjan
        if target_name.startswith(("DEF", "MCH", "ORG")):
            resolved = resolve_controller(rig, target_name, visited)
            if resolved:
                return resolved
            continue

        # Annars har vi hittat en controller
        return target_name

    return None



def export_bone_mappings():
    rig = bpy.data.objects.get("rig")
    if not rig or rig.type != "ARMATURE":
        raise RuntimeError("Hittar ingen armature som heter 'rig'")

    data = {}
    for bone in rig.data.bones:
        if not bone.name.startswith("DEF"):
            continue

        controller = resolve_controller(rig, bone.name)

        entry = {
            "parent": bone.parent.name if bone.parent else None,
            "head": list(bone.head_local),
            "tail": list(bone.tail_local),
            "controller": controller
        }

        data[bone.name] = entry
        print(f"[MAP] {bone.name} → {controller}")

    project_root = Path(bpy.path.abspath("//.."))
    out_dir = project_root / "outputs" / "animation_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "def_to_ctrl_map.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"bones": data}, f, indent=2)

    print(f"[INFO] Exporterade {len(data)} DEF-bones till {out_path}")

export_bone_mappings()
