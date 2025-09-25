# dataset_pipeline/blender_export/new_render_dataset.py
import bpy, json, os, math, argparse, sys
from mathutils import Vector
from pathlib import Path

# ---------- CLI ----------
def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--actions", default="Walk,Trot,LeftCanter,RightCanter,LeftGallop,RightGallop,Jump,BASE,IdleBase,IdleRest")
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--cameras", type=int, default=12)
    p.add_argument("--cam_radius", type=float, default=0.0)
    p.add_argument("--cam_height", type=float, default=1.6)
    p.add_argument("--single_frame", action="store_true")
    p.add_argument("--root_bone", default="root")
    p.add_argument("--bone_prefix", default="DEF-")
    p.add_argument("--variant", choices=["flat","materials","both"], default="both")
    return p.parse_args(argv)

args = get_args()

# ---------- Outdir ----------
blend_path = Path(bpy.data.filepath) if bpy.data.filepath else Path.cwd()
project_dir = blend_path.parent.parent
outdir_abs = Path(args.outdir)
if not outdir_abs.is_absolute():
    outdir_abs = (project_dir / outdir_abs).resolve()
outdir_abs.mkdir(parents=True, exist_ok=True)
args.outdir = str(outdir_abs)

scene = bpy.context.scene

# ---------- Render setup ----------
def setup_render_engine(scene, engine="BLENDER_EEVEE_NEXT", res=1024):
    engine = (engine or "").upper()
    if engine in {"EEVEE", "BLENDER_EEVEE"}:
        engine = "BLENDER_EEVEE_NEXT"
    scene.render.engine = engine
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.image_settings.file_format = "PNG"

    # Extra Sun-lamps
    if engine in {"BLENDER_EEVEE_NEXT", "CYCLES"}:
        def add_sun(name, strength, location, rotation):
            sun = bpy.data.lights.new(name, 'SUN')
            sun.energy = strength
            so = bpy.data.objects.new(name, sun)
            bpy.context.collection.objects.link(so)
            so.location = location
            so.rotation_euler = rotation
            return so
        add_sun("FillSun_L", 3.0, (8, -8, 6), (math.radians(50), 0, math.radians(45)))
        add_sun("FillSun_R", 3.0, (-8, 8, 6), (math.radians(60), 0, math.radians(-45)))
        add_sun("BackSun", 2.5, (0, 0, 12), (math.radians(90), 0, 0))

setup_render_engine(scene, args.engine, args.res)

# Dölj rigg/widgets i render
for obj in bpy.data.objects:
    if obj.type == "ARMATURE" or obj.name.startswith(("WGT-", "CTRL-", "MCH-")):
        obj.hide_render = True

# ---------- Armature ----------
arm = bpy.data.objects.get("rig")
if not arm:
    raise RuntimeError("Ingen armature 'rig' hittades i scenen!")

root_pb = arm.pose.bones.get(args.root_bone)
if not root_pb:
    raise RuntimeError("Ingen bone 'root' hittades!")

# ---------- Meshfilter ----------
def mesh_bound_to_arm(o, arm_obj):
    if o.parent == arm_obj:
        return True
    for m in o.modifiers:
        if m.type == "ARMATURE" and getattr(m, "object", None) == arm_obj:
            return True
    return False

dataset_meshes = []
for o in bpy.data.objects:
    if o.type != "MESH": continue
    if mesh_bound_to_arm(o, arm):
        o.hide_render = False
        try: o.hide_set(False)
        except: pass
        dataset_meshes.append(o.name)
    else:
        o.hide_render = True
        try: o.hide_set(True)
        except: pass

original_materials = {
    name: [m for m in (bpy.data.objects[name].data.materials[:] if bpy.data.objects.get(name) else [])]
    for name in dataset_meshes
}

def restore_materials():
    for name, mats in original_materials.items():
        o = bpy.data.objects.get(name)
        if not o: continue
        o.data.materials.clear()
        for m in mats:
            o.data.materials.append(m)

def make_flat_material():
    mat = bpy.data.materials.get("Dataset_Flat")
    if mat is None:
        mat = bpy.data.materials.new("Dataset_Flat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.6
        nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat

def apply_flat_material(mesh_names):
    mat = make_flat_material()
    for name in mesh_names:
        o = bpy.data.objects.get(name)
        if not o: continue
        o.data.materials.clear()
        o.data.materials.append(mat)

# ---------- Kameror ----------
def setup_cameras(act_name, center, radius, cam_height, num_cams=12):
    # rensa gamla kameror
    for o in list(bpy.data.objects):
        if o.name.startswith(f"{act_name}_Cam_") or o.name == f"{act_name}_Target":
            try: bpy.data.objects.remove(o, do_unlink=True)
            except: pass

    tgt = bpy.data.objects.new(f"{act_name}_Target", None)
    bpy.context.collection.objects.link(tgt)
    tgt.location = center

    cameras = []
    cam_index = 0
    heights = [cam_height, cam_height + radius*0.5, cam_height + radius]
    for h in heights:
        for i in range(num_cams):
            ang = (2*math.pi)*i/num_cams
            x = center.x + radius*math.cos(ang)
            y = center.y + radius*math.sin(ang)
            cam_data = bpy.data.cameras.new(f"{act_name}_Cam_{cam_index:02d}")
            cam_data.clip_start = 0.01; cam_data.clip_end = 2000.0; cam_data.lens = 50.0
            cam_obj = bpy.data.objects.new(f"{act_name}_Cam_{cam_index:02d}", cam_data)
            bpy.context.collection.objects.link(cam_obj)
            cam_obj.location = (x, y, h)
            con = cam_obj.constraints.new(type='TRACK_TO'); con.target = tgt
            con.track_axis = 'TRACK_NEGATIVE_Z'; con.up_axis = 'UP_Y'
            cameras.append(cam_obj)
            cam_index += 1
    return cameras

# ---------- Main ----------
variants = [args.variant] if args.variant != "both" else ["flat", "materials"]
actions = [s.strip() for s in args.actions.split(",") if s.strip()]

for act_name in actions:
    act = bpy.data.actions.get(act_name)
    assert act, f"Action '{act_name}' saknas"
    arm.animation_data_create(); arm.animation_data.action = act

    outdir_action = outdir_abs / act_name.lower()
    labels_dir = outdir_action / "labels"
    image_dirs = {v: outdir_action / "images" / v for v in variants}
    for d in image_dirs.values(): d.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    f_start, f_end = map(int, act.frame_range)
    frame_range = [f_start] if args.single_frame else range(f_start, f_end+1)

    # EDIT-mode: samla statisk info
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for b in arm.data.edit_bones:
        if not b.name.startswith("DEF"): continue
        edit_bones[b.name] = {
            "roll": float(b.roll),
            "head": [b.head.x, b.head.y, b.head.z],
            "tail": [b.tail.x, b.tail.y, b.tail.z],
        }
    bpy.ops.object.mode_set(mode="OBJECT")

    # setup cameras
    scene.frame_set(f_start)
    bone_pts = [arm.matrix_world @ pb.head for pb in arm.pose.bones if pb.name.startswith("DEF")] + \
               [arm.matrix_world @ pb.tail for pb in arm.pose.bones if pb.name.startswith("DEF")]
    center = sum(bone_pts, Vector((0,0,0))) / len(bone_pts) if bone_pts else arm.matrix_world.translation
    maxdist = max((p-center).length for p in bone_pts) if bone_pts else 4.0
    radius_auto = max(2.0, maxdist*3.5)
    radius = args.cam_radius if args.cam_radius > 0 else radius_auto
    cam_z = center.z + args.cam_height

    cameras = setup_cameras(act_name, center, radius, cam_z, args.cameras)

    for frame in frame_range:
        scene.frame_set(frame)

        # Root-matris
        root_mat_world = arm.matrix_world @ root_pb.matrix

        bones_payload = {}
        for pb in arm.pose.bones:
            if not pb.name.startswith("DEF"): continue
            static = edit_bones.get(pb.name, {})
            mat_world = arm.matrix_world @ pb.matrix
            mat_rel_root = root_mat_world.inverted() @ mat_world
            mat_pose = arm.matrix_world.inverted() @ mat_world
            rest = pb.bone.matrix_local
            mat_basis = rest.inverted() @ mat_pose

            bones_payload[pb.name] = {
                "roll": static.get("roll", 0.0),
                "head_edit": static.get("head"),
                "tail_edit": static.get("tail"),
                "matrix_world": [list(row) for row in mat_world],
                "matrix_rel_root": [list(row) for row in mat_rel_root],
                "matrix_basis": [list(row) for row in mat_basis],
            }

        # Spara JSON
        payload = {
            "action": act_name,
            "frame": frame,
            "armature": arm.name,
            "root_matrix_world": [list(row) for row in root_mat_world],
            "bones": bones_payload,
        }
        outpath = labels_dir / f"{frame:06d}.json"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Rendera bilder
        for ci, cam in enumerate(cameras):
            scene.camera = cam
            base_name = f"f{frame:05d}_c{ci:02d}"
            for variant in variants:
                if variant == "flat":
                    apply_flat_material(dataset_meshes)
                else:
                    restore_materials()
                images_dir = image_dirs[variant]
                png_path = images_dir / (base_name + ".png")
                scene.render.filepath = str(png_path)
                bpy.ops.render.render(write_still=True)

        print(f"[render_dataset] Frame {frame} klar för {act_name}")

    print(f"[render_dataset] Klart för action {act_name} → {outdir_action}")
