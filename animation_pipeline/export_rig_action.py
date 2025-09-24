# animation_pipeline/export_rig_action.py
import bpy
import json
import sys
from pathlib import Path


def export_action(action_name: str):
    project_root = Path(bpy.path.abspath("//.."))
    export_name = f"{action_name}_rigexport_rootrel"
    outdir = project_root / "outputs" / "rig_action_exports" / export_name
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Hämta armature ---
    arm = bpy.data.objects.get("rig")
    if arm is None:
        raise RuntimeError("Ingen armature 'rig' i scenen!")

    # --- Hämta action ---
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise RuntimeError(f"Hittar ingen action '{action_name}' i blend-filen.")

    arm.animation_data_create()
    arm.animation_data.action = action
    scene = bpy.context.scene

    # --- Root bone ---
    root_pb = arm.pose.bones.get("root")
    if not root_pb:
        raise RuntimeError("Ingen bone 'root' hittades i riggen!")

    f_start, f_end = map(int, action.frame_range)

    # --- EDIT-mode: samla statisk info ---
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for b in arm.data.edit_bones:
        if not b.name.startswith("DEF"):
            continue
        edit_bones[b.name] = {
            "roll": float(b.roll),
            "head": [b.head.x, b.head.y, b.head.z],
            "tail": [b.tail.x, b.tail.y, b.tail.z],
        }
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- Frame loop ---
    for f in range(f_start, f_end + 1):
        scene.frame_set(f)

        # Root-matris i world space
        root_mat_world = arm.matrix_world @ root_pb.matrix

        bones_payload = {}
        for pb in arm.pose.bones:
            if not pb.name.startswith("DEF"):
                continue

            static = edit_bones.get(pb.name, {})

            # World & root-relativ matris
            mat_world = arm.matrix_world @ pb.matrix
            mat_rel_root = root_mat_world.inverted() @ mat_world

            # Basis (rest-relative) matris
            mat_pose = arm.matrix_world.inverted() @ mat_world
            rest = pb.bone.matrix_local
            mat_basis = rest.inverted() @ mat_pose

            bones_payload[pb.name] = {
                # Statisk rigg-info
                "roll": static.get("roll", 0.0),
                "head_edit": static.get("head"),
                "tail_edit": static.get("tail"),

                # Matriser
                "matrix_world": [list(row) for row in mat_world],
                "matrix_rel_root": [list(row) for row in mat_rel_root],
                "matrix_basis": [list(row) for row in mat_basis],  # 🔑 exportera basis
            }

        payload = {
            "action": export_name,
            "frame": f - f_start + 1,
            "armature": arm.name,
            "root_matrix_world": [list(row) for row in root_mat_world],
            "bones": bones_payload,
        }

        outpath = outdir / f"{f:06d}.json"
        with open(outpath, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    print(f"[export_rig_action] Exporterade {f_end - f_start + 1} frames "
          f"(DEF: world, root-relativ & basis) → {outdir}")


# =====================
#  MAIN
# =====================
if __name__ == "__main__":
    action_name = None
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        if idx + 1 < len(sys.argv):
            action_name = sys.argv[idx + 1]
    if not action_name:
        raise RuntimeError("Ange en action med -- ActionName")

    export_action(action_name)
