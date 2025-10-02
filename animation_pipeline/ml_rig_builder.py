# animation_pipeline/ml_rig_builder.py
import bpy
import json

def build_ml_rig(json_path: str):
    """Bygg ML_rig från JSON med head/tail/roll/parent. Ignorerar constraints."""

    # Läs in JSON-data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bones_data = data["bones"]

    # Radera gammal ML_rig om den finns
    if "ML_rig" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["ML_rig"], do_unlink=True)

    # Skapa ny armature
    arm_data = bpy.data.armatures.new("ML_rig_data")
    ml_rig = bpy.data.objects.new("ML_rig", arm_data)
    bpy.context.collection.objects.link(ml_rig)
    bpy.context.view_layer.objects.active = ml_rig
    ml_rig.select_set(True)

    ml_rig.show_in_front = True
    ml_rig.data.display_type = 'OCTAHEDRAL'

    # --- Skapa bones ---
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = ml_rig.data.edit_bones

    # Först skapa alla bones
    for name, info in bones_data.items():
        bone = edit_bones.new(name)
        bone.head = tuple(info["head"])
        bone.tail = tuple(info["tail"])
        bone.roll = info.get("roll", 0.0)

    # Lägg till neutral root
    root = edit_bones.new("root")
    root.head = (0, 0, 0)
    root.tail = (0, 0, 0.5)

    # Koppla parent enligt JSON
    for name, info in bones_data.items():
        if name == "root":
            continue
        edit_bones[name].parent = root
        edit_bones[name].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[INFO] ML_rig skapad med {len(bones_data)} DEF-bones (+ root, hierarki från parent-fältet)")
    return ml_rig

