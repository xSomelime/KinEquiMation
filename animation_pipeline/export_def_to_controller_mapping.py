import bpy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
blend_file = PROJECT_ROOT / "assets" / "horse_model_rigifyDEF_v1.blend"
armature_name = "rig"
out_path = PROJECT_ROOT / "outputs" / "animation_pipeline" / "controller_mapping.json"

# --- Manuell lista från Amanda ---
HARDCODED_MAP = {
    "DEF-spine.001": "tweak_spine.001",
    "DEF-spine.002": "tweak_spine.002",
    "DEF-spine.003": "tweak_spine.003",
    "DEF-spine.004": "tweak_spine.004",
    "DEF-spine.005": "tweak_spine.005",
    "DEF-spine.006": "tweak_spine.006",

    "DEF-tail.001": "tweak_tail.001",
    "DEF-tail.002": "tweak_tail.002",
    "DEF-tail.003": "tweak_tail.003",
    "DEF-tail.004": "tweak_tail.004",
    "DEF-tail.005": "tweak_tail.005",

    "DEF-pelvis.L": "",
    "DEF-thigh.L": "thigh_tweak.L",
    "DEF-thigh.L.001": "thigh_tweak.L.001",
    "DEF-lower_leg.L": "lower_leg_tweak.L",
    "DEF-lower_leg.L.001": "lower_leg_tweak.L.001",
    "DEF-hind_foot.L": "hind_foot_heel_ik.L",
    "DEF-hind_foot.L.001": "hind_foot_tweak.L.001",
    "DEF-r_toe.L": "r_toe_ik.L",
    "DEF-r_hoof.L": "hind_foot_ik.L",

    "DEF-pelvis.R": "",
    "DEF-thigh.R": "thigh_tweak.R",
    "DEF-thigh.R.001": "thigh_tweak.R.001",
    "DEF-lower_leg.R": "lower_leg_tweak.R",
    "DEF-lower_leg.R.001": "lower_leg_tweak.R.001",
    "DEF-hind_foot.R": "hind_foot_heel_ik.R",
    "DEF-hind_foot.R.001": "hind_foot_tweak.R.001",
    "DEF-r_toe.R": "r_toe_ik.R",
    "DEF-r_hoof.R": "hind_foot_ik.R",

    "DEF-shoulder.L": "shoulder.L",
    "DEF-upper_arm.L": "upper_arm_ik.L",
    "DEF-upper_arm.L.001": "forearm_tweak.L",
    "DEF-forearm.L": "forearm_tweak.L.001",
    "DEF-forearm.L.001": "forefoot_tweak.L",
    "DEF-forefoot.L": "forefoot_heel_ik.L",
    "DEF-forefoot.L.001": "forefoot_tweak.L.001",
    "DEF-f_toe.L": "f_toe_tweak.L",
    "DEF-f_hoof.L": "forefoot_ik.L",
    "DEF-breast.L": "breast.L",

    "DEF-shoulder.R": "shoulder.R",
    "DEF-upper_arm.R": "upper_arm_ik.R",
    "DEF-upper_arm.R.001": "forearm_tweak.R",
    "DEF-forearm.R": "forearm_tweak.R.001",
    "DEF-forearm.R.001": "forefoot_tweak.R",
    "DEF-forefoot.R": "forefoot_heel_ik.R",
    "DEF-forefoot.R.001": "forefoot_tweak.R.001",
    "DEF-f_toe.R": "f_toe_tweak.R",
    "DEF-f_hoof.R": "forefoot_ik.R",
    "DEF-breast.R": "breast.R",

    "DEF-chest": "chest",

    "DEF-neck.001": "tweak_neck.001",
    "DEF-neck.002": "tweak_neck.002",
    "DEF-neck.003": "tweak_neck.003",
    "DEF-neck.004": "tweak_neck.004",

    "DEF-head": "head",
    "DEF-skull": "",
    "DEF-skull.L": "",
    "DEF-skull.R": "",

    "DEF-eye.L": "eye.L",
    "DEF-eye.R": "eye.R",
    "DEF-nose.L": "nose.L",
    "DEF-nose.R": "nose.R",

    "DEF-ear.L": "tweak_ear.L",
    "DEF-ear.L.001": "tweak_ear.L.001",
    "DEF-ear.R": "tweak_ear.R",
    "DEF-ear.R.001": "tweak_ear.R.001",

    "DEF-jaw": "tweak_jaw",
    "DEF-jaw.001": "tweak_jaw.001",

    "DEF-mane_base.01": "mane_base.01",
    "DEF-mane_base.02": "mane_base.02",
    "DEF-mane_base.03": "mane_base.03",
    "DEF-mane_base.04": "mane_base.04",
    "DEF-mane_base.05": "mane_base.05",
    "DEF-mane_base.06": "mane_base.06",

    "DEF-mane_top.01": "mane_top.01",
    "DEF-mane_top.02": "mane_top.02",
    "DEF-mane_top.03": "mane_top.03",
    "DEF-mane_top.04": "mane_top.04",
    "DEF-mane_top.05": "mane_top.05",
    "DEF-mane_top.06": "mane_top.06"
}

# --- Main ---
def main():
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))
    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise RuntimeError(f"Armature '{armature_name}' hittades inte i scenen.")

    mapping = {}
    missing = []

    for bone in arm_obj.pose.bones:
        if bone.name.startswith("DEF"):
            best = HARDCODED_MAP.get(bone.name, "")
            if best:
                mapping[bone.name] = {"controller": best}
                print(f"[MAP] {bone.name} → {best}")
            else:
                missing.append(bone.name)
                mapping[bone.name] = {"controller": ""}
                print(f"[MISS] {bone.name} → ingen controller")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Exporterat mapping till {out_path}")
    if missing:
        print("[WARN] Dessa DEF-bones saknar controllers:")
        for name in missing:
            print("  -", name)


if __name__ == "__main__":
    main()
