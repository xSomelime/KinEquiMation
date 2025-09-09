import os
from mmpose.apis import init_pose_model, inference_topdown, vis_pose_result
from mmpose.structures import merge_data_samples
import mmcv

# === Konfig och checkpoint för häst HRNet-W32 ===
CONFIG = "ai/models/checkpoints/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py"
CHECKPOINT = "ai/models/checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth"

# === Skapa output-mapp om den inte finns ===
os.makedirs("ai/outputs/test_poses", exist_ok=True)

# === Ladda modellen ===
model = init_pose_model(CONFIG, CHECKPOINT, device="cuda:0")

# === Testa på en bild ===
img_path = "data/images/test_horse.png"  # byt till egen bild

image = mmcv.imread(img_path)
h, w = image.shape[:2]
horse_instance = [{'bbox': [0, 0, w, h]}]

results = inference_topdown(model, image, horse_instance)
data_sample = merge_data_samples(results)

vis_img = vis_pose_result(
    model,
    image,
    results,
    kpt_score_thr=0.3,
    radius=4,
    thickness=2,
    show=False
)

mmcv.imwrite(vis_img, "ai/outputs/test_poses/output_pose.png")
print("✅ Klar! Resultatet sparat i ai/outputs/test_poses/output_pose.png")
