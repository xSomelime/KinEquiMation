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


# --- Constraint helpers (oförändrade) ---
def add_leg_ik_and_limit(arm, hoof_name, chain_start, chain_count, ground_obj=None):
    """Skapar IK från chain_start → hoof, och lägger limit så att hoven inte går under z=0.
       Om ground_obj anges används den som referens för markkontakt."""
    hoof = arm.pose.bones.get(hoof_name)
    start = arm.pose.bones.get(chain_start)
    if not hoof or not start:
        print(f"[WARN] Kunde inte skapa IK för {hoof_name}")
        return

    # IK
    ik = start.constraints.new('IK')
    ik.target = arm
    ik.subtarget = hoof.name
    ik.chain_count = chain_count

    # Limit Z
    limit = hoof.constraints.new('LIMIT_LOCATION')
    limit.use_min_z = True
    limit.min_z = 0.0
    limit.owner_space = 'LOCAL'
    print(f"[INFO] IK + Limit Location lagt på {hoof_name} (Z ≥ 0)")

    if ground_obj:
        floor = hoof.constraints.new('FLOOR')
        floor.target = ground_obj
        floor.use_sticky = True
        floor.sticky_radius = 0.05
        floor.offset = 0.0
        print(f"[INFO] Floor-constraint lagt på {hoof_name} mot {ground_obj.name}")


def add_constraints(ml_rig):
    """Lägg till IK på alla fyra ben och parenta head → skull."""
    bpy.ops.object.mode_set(mode='POSE')

    add_leg_ik_and_limit(ml_rig, hoof_name="DEF-f_hoof.L",
                         chain_start="DEF-forearm.L", chain_count=5)
    add_leg_ik_and_limit(ml_rig, hoof_name="DEF-f_hoof.R",
                         chain_start="DEF-forearm.R", chain_count=5)
    add_leg_ik_and_limit(ml_rig, hoof_name="DEF-r_hoof.L",
                         chain_start="DEF-lower_leg.L", chain_count=6)
    add_leg_ik_and_limit(ml_rig, hoof_name="DEF-r_hoof.R",
                         chain_start="DEF-lower_leg.R", chain_count=6)

    # Head → Skull
    bpy.ops.object.mode_set(mode='EDIT')
    eb = ml_rig.data.edit_bones
    if "DEF-head" in eb and "DEF-skull" in eb:
        eb["DEF-head"].parent = eb["DEF-skull"]
        print("[INFO] DEF-head parented till DEF-skull")
    bpy.ops.object.mode_set(mode='POSE')

    print("[INFO] Alla constraints tillagda")
