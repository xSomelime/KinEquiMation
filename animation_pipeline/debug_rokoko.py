import bpy
import addon_utils

def dump_props(label, obj):
    print(f"\n=== Alla Rokoko-relaterade properties i {label} ===")
    for prop in dir(obj):
        if "rokoko" in prop.lower():
            try:
                value = getattr(obj, prop)
            except Exception as e:
                value = f"<error: {e}>"
            print(f"{prop}: {value}")

print("=== Alla Rokoko-operators i bpy.ops ===")
for op in dir(bpy.ops):
    if "rokoko" in op.lower():
        print(f"- {op}")

# Om bpy.ops.rokoko finns, lista även dess sub-operators
if hasattr(bpy.ops, "rokoko"):
    print("\n=== Sub-operators i bpy.ops.rokoko ===")
    for sub in dir(bpy.ops.rokoko):
        print(f"- rokoko.{sub}")

# Kolla vanliga platser för addon-properties
dump_props("bpy.context.scene", bpy.context.scene)
dump_props("bpy.context.window_manager", bpy.context.window_manager)
dump_props("bpy.context.preferences", bpy.context.preferences)

# Kolla aktiva objekt och armature
if bpy.context.object:
    dump_props(f"Object {bpy.context.object.name}", bpy.context.object)
    if bpy.context.object.type == "ARMATURE":
        dump_props(f"Armature {bpy.context.object.data.name}", bpy.context.object.data)

# Lista addons
print("\n=== Alla Rokoko-relaterade addons ===")
for addon in bpy.context.preferences.addons.keys():
    if "rokoko" in addon.lower():
        print(f"- {addon}")

# Lista moduler
print("\n=== Alla Rokoko-moduler via addon_utils ===")
for mod in addon_utils.modules():
    if "rokoko" in mod.__name__.lower():
        print(f"- {mod.__name__}")

# Extra: kolla bpy.types (klassregistreringar)
print("\n=== Alla bpy.types med 'rokoko' i namnet ===")
for t in dir(bpy.types):
    if "rokoko" in t.lower():
        print(f"- {t}")
