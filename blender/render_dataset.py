# blender/render_dataset.py
import bpy, json, os, math, argparse, sys
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

# ---------- CLI ----------
def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--actions", default="Walk,Trot,LeftCanter,RightCanter,LeftGallop,RightGallop,Jump,BASE")
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT")   # Blender 4.4
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--cameras", type=int, default=12)
    p.add_argument("--cam_radius", type=float, default=0.0, help="0 = auto från bones")
    p.add_argument("--cam_height", type=float, default=1.6)
    p.add_argument("--root_bone", default="root")
    p.add_argument("--bone_prefix", default="DEF-")
    p.add_argument("--axis_scale", type=float, default=0.25)
    p.add_argument("--variant", choices=["flat","materials","both"], default="flat",
                   help="Vilken typ av bilder som ska renderas")
    return p.parse_args(argv)

args = get_args()

# ---------- Outdir relativt projektrot ----------
blend_path = Path(bpy.data.filepath) if bpy.data.filepath else Path.cwd()
blend_dir = blend_path.parent
project_dir = blend_dir.parent if blend_dir.name.lower() == "assets" else blend_dir
outdir_abs = Path(args.outdir)
if not outdir_abs.is_absolute():
    outdir_abs = (project_dir / outdir_abs).resolve()
outdir_abs.mkdir(parents=True, exist_ok=True)
args.outdir = str(outdir_abs)

# ---------- Scene ----------
scene = bpy.context.scene

# --- Extra Sun-lampor för "outdoor"-känsla ---
if args.engine in {"BLENDER_EEVEE_NEXT", "CYCLES"}:
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

engine = (args.engine or "").upper()
if engine in {"EEVEE", "BLENDER_EEVEE"}:
    engine = "BLENDER_EEVEE_NEXT"
scene.render.engine = engine
scene.render.resolution_x = args.res
scene.render.resolution_y = args.res
scene.render.image_settings.file_format = "PNG"

# Dölj rigg/widgets i render (mesh lämnas synligt)
for obj in bpy.data.objects:
    if obj.type == "ARMATURE" or obj.name.startswith(("WGT-", "CTRL-", "MCH-")):
        obj.hide_render = True

# ---------- Armature ----------
def pick_armature():
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    assert arms, "Ingen ARMATURE hittades i scenen."
    def score(a):
        mesh_cnt = 0
        for o in bpy.data.objects:
            if o.type != "MESH": continue
            if o.parent == a: mesh_cnt += 1; continue
            for m in o.modifiers:
                if m.type=="ARMATURE" and getattr(m,"object",None)==a:
                    mesh_cnt+=1; break
        def_bones = sum(1 for pb in a.pose.bones if pb.name.startswith("DEF-"))
        return (mesh_cnt, def_bones)
    best = max(arms, key=score)
    print(f"[render_dataset] Armature vald: {best.name}")
    return best
arm = pick_armature()

# ---------- Benfilter ----------
ALLOW_PREFIXES = ["spine","neck","head","skull","jaw","shoulder","upper_arm","forearm","forefoot","f_toe","f_hoof",
    "chest","abdomen","pelvis","thigh","lower_leg","hind_foot","r_toe","r_hoof","hip","tail"]
has_def = any(pb.name.startswith(args.bone_prefix) for pb in arm.pose.bones)
export_bone_names=set()
for pb in arm.pose.bones:
    n=pb.name; l=n.lower()
    if "mane" in l: continue
    if l.startswith("ear"): export_bone_names.add(n); continue
    if has_def:
        if n.startswith(args.bone_prefix): export_bone_names.add(n)
    else:
        if any(l.startswith(pfx) for pfx in ALLOW_PREFIXES): export_bone_names.add(n)
assert export_bone_names,"Inga exporterbara ben hittades."

# ---------- Root ----------
def choose_root_bone(arm_obj, desired_name, bone_candidates):
    pb = arm_obj.pose.bones.get(desired_name)
    if pb: return pb, f"bone:{desired_name}"
    for name in ["hips","hip","pelvis","chest","spine","spine.001","abdomen"]:
        cand=arm_obj.pose.bones.get(name)
        if cand: return cand,f"bone:{name}"
    return None,"armature_object"
root_pb, root_method = choose_root_bone(arm,args.root_bone,export_bone_names)
print(f"[render_dataset] Root-val: {root_method}")

# ---------- Meshfilter ----------
def mesh_bound_to_arm(o,arm_obj):
    if o.parent==arm_obj: return True
    for m in o.modifiers:
        if m.type=="ARMATURE" and getattr(m,"object",None)==arm_obj: return True
    return False
dataset_meshes=[]
for o in bpy.data.objects:
    if o.type!="MESH": continue
    if mesh_bound_to_arm(o,arm):
        o.hide_render=False
        try:o.hide_set(False)
        except:pass
        dataset_meshes.append(o.name)
    else:
        o.hide_render=True
        try:o.hide_set(True)
        except:pass
print(f"[render_dataset] Aktiva meshar: {dataset_meshes}")

# ---------- Original materials ----------

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


# ---------- Flat material ----------
def make_flat_material():
    mat=bpy.data.materials.get("Dataset_Flat")
    if mat is None:
        mat=bpy.data.materials.new("Dataset_Flat")
        mat.use_nodes=True
        nt=mat.node_tree
        for n in list(nt.nodes): nt.nodes.remove(n)
        out=nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf=nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value=(0.8,0.8,0.8,1.0)
        if "Specular" in bsdf.inputs: bsdf.inputs["Specular"].default_value=0.0
        if "Roughness" in bsdf.inputs: bsdf.inputs["Roughness"].default_value=0.6
        nt.links.new(bsdf.outputs[0],out.inputs[0])
    return mat

def apply_flat_material(mesh_names):
    mat=make_flat_material()
    for name in mesh_names:
        o=bpy.data.objects.get(name)
        if not o: continue
        o.data.materials.clear()
        o.data.materials.append(mat)

# ---------- Helpers ----------
def bone_world_head(a,pb): return (a.matrix_world @ pb.head)
def bone_world_tail(a,pb): return (a.matrix_world @ pb.tail)
def bone_world_matrix(a,pb): return (a.matrix_world @ pb.matrix)
def get_intrinsics(scene,cam_obj):
    r=scene.render
    res_x,res_y=r.resolution_x,r.resolution_y
    f_mm=cam_obj.data.lens
    sw,sh=cam_obj.data.sensor_width,cam_obj.data.sensor_height
    fx=f_mm/sw*res_x; fy=f_mm/sh*res_y
    cx=res_x*0.5; cy=res_y*0.5
    K=[[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]]
    return {"width":res_x,"height":res_y,"fx":fx,"fy":fy,"cx":cx,"cy":cy,"K":K}
def project_point(scene,cam_obj,world_co):
    ndc=world_to_camera_view(scene,cam_obj,world_co)
    x_ndc,y_ndc=float(ndc.x),float(ndc.y)
    w,h=scene.render.resolution_x,scene.render.resolution_y
    u=x_ndc*w; v=(1.0-y_ndc)*h
    p_cam=cam_obj.matrix_world.inverted()@world_co
    depth=float(-p_cam.z)
    in_frame=(0<=x_ndc<=1 and 0<=y_ndc<=1 and depth>0)
    return (u,v),(x_ndc,y_ndc),in_frame,depth

# ---------- Main ----------
variants=[args.variant] if args.variant!="both" else ["flat","materials"]
actions=[s.strip() for s in args.actions.split(",") if s.strip()]
assert actions,"Ange minst en action i --actions."

for act_name in actions:
    act=bpy.data.actions.get(act_name)
    assert act,f"Action '{act_name}' saknas i .blend."
    arm.animation_data_create(); arm.animation_data.action=act

    outdir_action=os.path.join(args.outdir, act_name.lower())
    labels_dir=os.path.join(outdir_action,"labels")
    os.makedirs(labels_dir,exist_ok=True)
    image_dirs={v: os.path.join(outdir_action,"images",v) for v in variants}
    for d in image_dirs.values(): os.makedirs(d,exist_ok=True)

    # setup frames
    f_start,f_end=map(int,act.frame_range)

    # center from bones
    scene.frame_set(f_start)
    bone_pts=[bone_world_head(arm,pb) for b in export_bone_names if (pb:=arm.pose.bones.get(b))]+\
             [bone_world_tail(arm,pb) for b in export_bone_names if (pb:=arm.pose.bones.get(b))]
    if bone_pts:
        minv=Vector((min(p.x for p in bone_pts),min(p.y for p in bone_pts),min(p.z for p in bone_pts)))
        maxv=Vector((max(p.x for p in bone_pts),max(p.y for p in bone_pts),max(p.z for p in bone_pts)))
        center=(minv+maxv)*0.5
        maxdist=max((p-center).length for p in bone_pts)
    else:
        center=arm.matrix_world.translation; maxdist=4.0
    radius_auto=max(2.0,maxdist*1.8)
    radius=args.cam_radius if args.cam_radius>0 else radius_auto
    cam_z=center.z+args.cam_height

    # clear old cams
    for o in list(bpy.data.objects):
        if o.name.startswith(f"{act_name}_Cam_") or o.name==f"{act_name}_Target":
            try:bpy.data.objects.remove(o,do_unlink=True)
            except:pass

    # target empty
    tgt=bpy.data.objects.new(f"{act_name}_Target",None)
    bpy.context.collection.objects.link(tgt); tgt.location=center

    # cameras
    cameras=[]
    for i in range(args.cameras):
        ang=(2*math.pi)*i/args.cameras
        x=center.x+radius*math.cos(ang)
        y=center.y+radius*math.sin(ang)
        cam_data=bpy.data.cameras.new(f"{act_name}_Cam_{i:02d}")
        cam_data.clip_start=0.01; cam_data.clip_end=2000.0; cam_data.lens=50.0
        cam_obj=bpy.data.objects.new(f"{act_name}_Cam_{i:02d}",cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location=(x,y,cam_z)
        con=cam_obj.constraints.new(type='TRACK_TO'); con.target=tgt
        con.track_axis='TRACK_NEGATIVE_Z'; con.up_axis='UP_Y'
        cameras.append(cam_obj)

    print(f"[render_dataset] {act_name}: center={tuple(round(v,3) for v in center)}, "
          f"radius={round(radius,3)}, cam_z={round(cam_z,3)}, cams={len(cameras)}")

    # frames
    for frame in range(f_start,f_end+1):
        scene.frame_set(frame)
        for ci,cam in enumerate(cameras):
            scene.camera=cam
            intr=get_intrinsics(scene,cam)
            cam_mw=cam.matrix_world
            cam_extr={"matrix_world":[[float(v) for v in row] for row in cam_mw]}
            # baseline payload
            bones_payload={}
            for bname in export_bone_names:
                pb=arm.pose.bones.get(bname)
                if not pb: continue
                head_w=bone_world_head(arm,pb); tail_w=bone_world_tail(arm,pb)
                uv_h,_,in_h,depth_h=project_point(scene,cam,head_w)
                uv_t,_,in_t,depth_t=project_point(scene,cam,tail_w)
                bones_payload[bname]={"parent":pb.parent.name if pb.parent else None,
                                      "uv":[uv_h[0],uv_h[1]],"in_frame":in_h,
                                      "uv_tail":[uv_t[0],uv_t[1]],"in_frame_tail":in_t,
                                      "depth_cam":depth_h}
            base_name=f"f{frame:05d}_c{ci:02d}"
            # render for each variant
            for variant in variants:
                if variant == "flat":
                    apply_flat_material(dataset_meshes)
                else:  # materials
                    restore_materials()

                images_dir = image_dirs[variant]
                png_path = os.path.join(images_dir, base_name + ".png")
                scene.render.filepath = png_path
                bpy.ops.render.render(write_still=True)

            # write json once
            json_path=os.path.join(labels_dir,base_name+".json")
            payload={"action":act_name,"frame":frame,
                     "image_size":[scene.render.resolution_x,scene.render.resolution_y],
                     "camera":cam.name,"camera_intrinsics":intr,"camera_extrinsics":cam_extr,
                     "armature":arm.name,"bones":bones_payload}
            with open(json_path,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)

print(f"[render_dataset] Klart! Dataset till: {args.outdir}")
