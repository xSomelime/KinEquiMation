# dataset_pipeline/blender_export/export_fbx.py

import bpy, os

# Vart filen ska sparas (relativt projektroten)
out = os.path.abspath("data/out.fbx")

# Exportera hela scenen som FBX
bpy.ops.export_scene.fbx(
    filepath=out,
    use_selection=False,
    bake_anim=True,
    add_leaf_bones=False,
    path_mode='AUTO',
    axis_forward='-Z', axis_up='Y'
)

print("Exported FBX to:", out)
