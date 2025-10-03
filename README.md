# KinEquiMation 🐎
**Markerless Motion Capture & Pose Estimation for Horses (2D → 3D)**

KinEquiMation är ett AI-baserat system som konverterar vanliga 2D-videor av hästar till 3D-riggad animation i Blender.  
Projektet eliminerar behovet av dyr motion capture-utrustning genom att kombinera **pose estimation**, **temporal 3D-lifting** och **retargeting**.

---

## 🚀 Projektstatus

### Fas 1 – Dataset ✅
- Syntetiskt dataset genererat i **Blender**.  
- Export av 3D ground truth (xyz + rotationer).  
- Dataset i **COCO Keypoints-format (68 kp)** klart.  

### Fas 2 – 2D Pose Estimation ✅
- Modell: **MMPose (HRNet-W32, animal-pretrain)**.  
- Tränad på syntetiska frames.  
- Overlay-script verifierar prediktionerna.  

### Fas 3 – 3D Lifting ✅
- Temporal modell inspirerad av **VideoPose3D**.  
- Tränad på syntetiska data.  
- `lifter_preds` exporteras och kan importeras till Blender.  

### Fas 4 – Retargeting ✅
- Python-script bygger **ML_rig** (kopierar DEF-bones utan constraints).  
- Mappning **DEF → controllers** möjliggör retargeting i Blender.  
- Retargeting görs med **Rokoko**.  
- Därefter kan animationen **justeras direkt på riggen** som retargetats till.  

---
KinEquiMation/
│
├── animation_pipeline/         # Blender Python scripts for retargeting
│   ├── build_apply_ml_rig.py       # Apply ML_rig (no constraints) from JSON
│   ├── export_def_to_controller_mapping.py  # DEF → controller mapping for retargeting
│   ├── export_skeleton_edges_from_rig.py    # Export skeleton edges for dataset building
│   ├── ml_rig_builder.py          # Build ML_rig from DEF bones
│   └── rokoko_retargerer.py       # Retarget ML_rig → rig using Rokoko
│
├── assets/                    # Static assets (rigs, models, etc.)
│   └── horse_model_rigifyDEF_v1.blend
│
├── dataset_pipeline/          # Scripts for dataset generation
│   ├── blender_export/            # Export per-frame metadata & renders from Blender
│   ├── build_dataset/             # Convert metadata to COCO-style datasets
│   └── data/                      # Final datasets and dataset exports
│
├── model_pipeline/            # Training & evaluation of ML models
│   ├── checkpoints/               # Saved model weights
│   ├── configs/                   # Training configs (pose/lifter)
│   ├── datasets/                  # Dataset loaders for PyTorch
│   ├── evaluation/                # Overlay, JSON/Blender export, visualization
│   ├── models/                    # PyTorch model definitions (pose, lifter)
│   └── training/                  # Training loops for pose & lifter models
│
├── outputs/                   # Experiment results and artifacts
│   ├── animation_pipeline/        # Blender animation exports
│   ├── hrnet_test_run/            # HRNet test outputs
│   ├── lifter_preds/              # 3D lifter predictions (JSON, npy)
│   ├── lifter_runs/               # Lifter training runs (logs, checkpoints)
│   ├── overlays_hrnet/            # Visual overlays from 2D predictions
│   └── overlays_lifter3d/         # Visual overlays from 3D lifter
│
├── venv/                     # Local Python virtual environment (ignored)
│
├── .gitignore
├── README.md
├── requirements.txt          # Dependencies for local environment
└── requirements-wsl.txt      # Dependencies for WSL2/Docker environment



## ⚙️ Tech Stack  

### Blender Python API ✅  
- Data generation  
- Rigging support  

### PyTorch ✅  
- Model training (pose + 3D lifter)  

### MMPose ✅  
- 2D keypoint detection  

### OpenCV ✅  
- Video processing  

### Rokoko (Blender) ✅  
- Retargeting DEF → controllers  

---

## 📊 MVP Goals ✅  

### Input  
- Simple horse video (clear background)  

### Output  
- 3D keypoints → imported into Blender  

### Retargeting  
- Retargeting to rig via ML_rig + Rokoko  
- Animation can be further adjusted directly on the retargeted rig  

---

## 🔮 Future Development ⏳  

### Kinematic Constraints  
- Bone length  
- Joint limits  
- Gait logic  

### Motion Smoothing  
- One-Euro filter  
- Other filtering methods  

### Fine-Tuning  
- Train 2D model on annotated real frames  

### Multi-Species Support  
- Extend to other quadrupeds  

### Real-Time Inference  
- Live 2D → 3D from webcam  

### Animator Tools  
- Blender/Maya plugin for animation workflows  

### Neural Motion Retargeting  
- Replace heuristic IK rules with ML-based methods  
