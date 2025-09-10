🐎 KinEquiMation

KinEquiMation is a project for training and using a 2D→3D horse pose estimation model, based on MMPose (HRNet) and a custom synthetic dataset generated in Blender.
It’s part of a school assignment, but also serves as the foundation for a future horse-themed game project.
Project Structure

```
📂 Project Structure
dataset_pipeline/       # Scripts for generating & building datasets
  ├── blender_export/   # Blender export scripts (FBX, labels, images, skeleton)
  ├── build_dataset/    # Scripts to merge & convert dataset (CSV, COCO JSON)
  └── debug/            # Sanity checks & visualization of dataset

model_pipeline/
  ├── configs/          # MMPose configs (HRNet, lifter, horse-specific configs)
  ├── checkpoints/      # Pretrained & fine-tuned weights (.pth) [ignored in Git]
  ├── training/         # Custom training loops (if needed)
  └── evaluation/       # Inference & visualization of model predictions

outputs/                # Final results (predictions, overlays, demo outputs, trained checkpoints)
assets/                 # Blender models (.blend, rigs, materials)

requirements.txt        # Python deps for Windows/Git Bash (CPU / no CUDA)
requirements-wsl.txt    # Python deps for WSL/Ubuntu (with CUDA support)
README.md




⚙️ Environment Setup

We use two environments:

    Windows + Git Bash (venv)
    Used for dataset generation (Blender scripts) and small utility scripts.

      Install Python 3.10 and dependencies:
  
      python -m venv venv
      . venv/Scripts/activate
      pip install -r requirements.txt




    WSL/Ubuntu (conda or venv)
    Used for training models with PyTorch + CUDA.

      Install dependencies with:
      
      conda create -n horse python=3.10
      conda activate horse
      pip install -r requirements-wsl.txt


⚠️ Note:

    Datasets (dataset_pipeline/data/…) and checkpoints (outputs/checkpoints/*.pth) are not tracked in Git (they’re listed in .gitignore).
    You can regenerate datasets using the Blender + build scripts.

    Pretrained HRNet weights are downloaded from OpenMMLab
     and stored in outputs/checkpoints/.

🚀 Workflow

    Generate dataset (Windows / Git Bash + Blender)

    Run dataset_pipeline/blender_export/*.py inside Blender to export frames, labels, skeleton.

    Run dataset_pipeline/build_dataset/export_dataset.py to create coco_synth_68.json and dataset.csv.

    Optionally, run dataset_pipeline/debug/sanity_check.py and viz_overlay_skeleton.py to validate dataset quality.

    Train the model (WSL/Ubuntu)

    Activate your environment.

    Train HRNet on the horse dataset with MMPose:

    mim train mmpose model_pipeline/configs/hrnet_w32_horse68_256x256.py


    Checkpoints will be saved in outputs/checkpoints/.

Evaluate & visualize

    Run model_pipeline/evaluation/inference.py on images or video:

      python model_pipeline/evaluation/inference.py \
        --config model_pipeline/configs/hrnet_w32_horse68_256x256.py \
        --checkpoint outputs/checkpoints/hrnet_w32_horse68_best.pth \
        --image data/images/test_horse.png \
        --out outputs/predictions/


    Use viz_overlay_predictions.py to compare predicted vs ground truth keypoints on your dataset.


✅ This way:

      Windows/Git Bash = dataset generation (with Blender).
  
      WSL/Ubuntu = model training with CUDA.
  
      Outputs/ = the actual results you want to use or show (keypoints, overlays, trained weights).

``` 
