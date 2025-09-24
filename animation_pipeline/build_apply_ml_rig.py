# animation_pipeline/build_apply_ml_rig.py
import bpy
import json
import sys
from pathlib import Path
import mathutils

project_root = Path(bpy.path.abspath("//.."))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from animation_pipeline.ml_rig_builder import build_ml_rig, add_constraints

DEBUG_BONES = {"DEF-spine.001", "DEF-forearm.L", "DEF-r_hoof.R"}


def anchor_root(ml_rig):
    root_bone = ml_rig.pose.bones.get("root")
    if not root_bone:
        print("[WARN] Ingen root hittades i ML_rig")
        return
    root_bone.lock_location = (True, True, True)
    root_bone.lock_rotation = (True, True, True)
    root_bone.lock_rotation_w = True
    print("[INFO] Root anchored & locked")


def compare_rigs(ref_rig_name="rig", test_rig_name="ML_rig"):
    ref = bpy.data.objects.get(ref_rig_name)
    test = bpy.data.objects.get(test_rig_name)
    if not ref or not test:
        print("[ERROR] Kunde inte hitta riggar att jämföra")
        return
    ref_bones = [b for b in ref.data.bones if b.name.startswith("DEF")]
    test_bones = [b for b in test.data.bones if b.name.startswith("DEF")]
    print(f"[DEBUG] {ref_rig_name}: {len(ref_bones)} DEF-bones, {test_rig_name}: {len(test_bones)} DEF-bones")


def prepare_and_apply(action_name=None, with_constraints=False):
    preds_root = project_root / "outputs" / "rig_action_exports"        #  <<---- senare "lifter_preds"
    json_path = project_root / "outputs" / "animation_pipeline" / "skeleton_edges_from_rig.json"

    ml_rig = build_ml_rig(str(json_path))
    anchor_root(ml_rig)
    compare_rigs("rig", "ML_rig")

    if action_name:
        action_dir = preds_root / action_name
        if not action_dir.exists():
            raise RuntimeError(f"Action-mapp saknas: {action_name}")
    else:
        action_dirs = sorted(preds_root.glob("*_rigexport_rootrel"))
        if not action_dirs:
            raise RuntimeError(f"Inga action-mappar i {preds_root}")
        action_dir = action_dirs[-1]

    print(f"[INFO] Använder action {action_dir.name}")

    new_action = bpy.data.actions.new(name=f"ML_{action_dir.name}")
    ml_rig.animation_data_create()
    ml_rig.animation_data.action = new_action
    print(f"[INFO] Ny action skapad: {new_action.name}")

    frame_files = sorted(action_dir.glob("*.json"))
    print(f"[INFO] Laddar {len(frame_files)} frames")

    for frame_idx, frame_file in enumerate(frame_files, start=1):
        with open(frame_file, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

        bones_data = frame_data.get("bones", {})
        bpy.context.scene.frame_set(frame_idx)

        root_mat_world = mathutils.Matrix(frame_data["root_matrix_world"])

        for bone_name, data in bones_data.items():
            if bone_name not in ml_rig.pose.bones:
                continue
            pb = ml_rig.pose.bones[bone_name]

            if "matrix_rel_root" in data and data["matrix_rel_root"]:
                mat_rel_root = mathutils.Matrix(data["matrix_rel_root"])

                # Gör om root-relativ till world
                mat_world = root_mat_world @ mat_rel_root

                # World → armature space
                mat_pose = ml_rig.matrix_world.inverted() @ mat_world

                # Korrigera mot benets restpose
                rest = pb.bone.matrix_local
                pb.matrix_basis = rest.inverted() @ mat_pose

                # Keyframes
                pb.keyframe_insert(data_path="location", frame=frame_idx)
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
                pb.keyframe_insert(data_path="scale", frame=frame_idx)

                if bone_name in DEBUG_BONES and frame_idx <= 3:
                    print(f"\n[DEBUG] Frame {frame_idx} Bone {bone_name}")
                    print("  Source (rel_root):", mat_rel_root)
                    print("  Rest local       :", rest)
                    print("  Applied basis    :", pb.matrix_basis)

        if frame_idx == 1:
            print(f"[DEBUG] Frame 1 applied ({len(bones_data)} DEF-bones)")

    print(f"[INFO] Animation '{new_action.name}' skapad med {len(frame_files)} frames.")

    if with_constraints:
        add_constraints(ml_rig)
        print("[INFO] Constraints applicerade på ML_rig")
    else:
        print("[INFO] Skippade constraints på ML_rig (debug-läge)")


if __name__ == "__main__":
    action_name = None
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        if idx + 1 < len(sys.argv):
            action_name = sys.argv[idx + 1]

    prepare_and_apply(action_name)
