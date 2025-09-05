# tools/viz_overlay_skeleton.py
# Ritar skeleton-overlay:
#  - parent→child (grön)
#  - head→tail (magenta)
#  - lokala axlar X/Y/Z (röd/grön/blå)
import os, json, argparse, glob, random
import cv2

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def draw_line(img, p1, p2, color, thick=2):
    cv2.line(img, p1, p2, color, thick, lineType=cv2.LINE_AA)

def draw_point(img, p, color, radius=4):
    cv2.circle(img, p, radius, color, -1, lineType=cv2.LINE_AA)

def overlay_one(image_path, json_path, out_path,
                alpha=0.6, thick=2, radius=4, draw_axes=True):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[!] kunde inte läsa bild: {image_path}")
        return
    data = load_json(json_path)
    bones = data.get("bones", {})

    canvas = img.copy()

    # A) parent→child (grön)
    for bname, b in bones.items():
        parent = b.get("parent")
        if not parent or parent not in bones:
            continue
        if not (b.get("in_frame") and bones[parent].get("in_frame")):
            continue
        u1, v1 = b["uv"]
        u2, v2 = bones[parent]["uv"]
        draw_line(canvas, (int(u1), int(v1)), (int(u2), int(v2)), (0, 255, 0), thick)

    # B) head→tail (magenta)
    if any(("uv_tail" in b) for b in bones.values()):
        for bname, b in bones.items():
            if not (b.get("in_frame") and b.get("in_frame_tail")):
                continue
            u1, v1 = b["uv"]; u2, v2 = b["uv_tail"]
            draw_line(canvas, (int(u1), int(v1)), (int(u2), int(v2)),
                      (255, 0, 255), max(1, thick-1))

    # C) axlar (röd/grön/blå)
    if draw_axes:
        for bname, b in bones.items():
            if not b.get("in_frame"):
                continue
            axes = b.get("axes")
            if not axes:
                continue
            u_h, v_h = b["uv"]
            p_h = (int(u_h), int(v_h))
            if axes.get("x_uv") and axes.get("x_in_frame"):
                u,v = axes["x_uv"]
                draw_line(canvas, p_h, (int(u), int(v)), (0,0,255), max(1, thick-1))
            if axes.get("y_uv") and axes.get("y_in_frame"):
                u,v = axes["y_uv"]
                draw_line(canvas, p_h, (int(u), int(v)), (0,255,0), max(1, thick-1))
            if axes.get("z_uv") and axes.get("z_in_frame"):
                u,v = axes["z_uv"]
                draw_line(canvas, p_h, (int(u), int(v)), (255,0,0), max(1, thick-1))

    # D) noder
    for bname, b in bones.items():
        if not b.get("in_frame"):
            continue
        u,v = b["uv"]; p = (int(u), int(v))
        color = (255,255,255)      # vit
        if bname.lower().startswith("ear"):
            color = (0,200,255)    # turkos
        elif bname.startswith("DEF-"):
            color = (0,165,255)    # orange
        draw_point(canvas, p, color, radius)

    out = cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset/final",
                    help="Rotmapp till datasetet (default: data/dataset/final)")
    ap.add_argument("--action", help="Filtrera på viss/vissa action(s), kommaseparerade")
    ap.add_argument("--suffix", default="_overlay", help="Suffix för sparade filer")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--no_axes", action="store_true", help="Stäng av ritning av axlar")
    ap.add_argument("--flat", action="store_true", help="Använd på dataset med flat-material (ingen texture)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Antal slumpmässiga bilder per action (0 = alla)")
    args = ap.parse_args()

    actions = []
    if args.action:
        actions = [a.strip().lower() for a in args.action.split(",")]

    # Gå igenom alla actions under root
    for action_dir in sorted(os.listdir(args.root)):
        if actions and action_dir.lower() not in actions:
            continue
        labels_dir = os.path.join(args.root, action_dir, "labels")
        if not os.path.isdir(labels_dir):
            continue

        json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
        if not json_files:
            continue

        # Om --sample används, välj slumpmässigt urval
        if args.sample and args.sample > 0:
            json_files = random.sample(json_files, min(args.sample, len(json_files)))

        for jp in json_files:
            variant = "materials" if not args.flat else "flat"
            png = jp.replace(os.sep + "labels" + os.sep,
                             os.sep + f"images{os.sep}{variant}{os.sep}")[:-5] + ".png"
            if not os.path.exists(png):
                continue

            overlays_dir = os.path.join(os.path.dirname(os.path.dirname(jp)), "overlays")
            os.makedirs(overlays_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(jp))[0]
            suffix = args.suffix + ("_flat" if args.flat else "")
            outp = os.path.join(overlays_dir, base_name + f"{suffix}.png")

            overlay_one(
                png, jp, outp,
                alpha=args.alpha,
                thick=args.thickness,
                radius=args.radius,
                draw_axes=(not args.no_axes)
            )
            print("✓", outp)

if __name__ == "__main__":
    main()
