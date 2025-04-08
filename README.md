## 3D Reconstruction Using NeRFs

**Neural Radiance Fields (NeRFs)** offer a powerful way to achieve photorealistic 3D scene reconstruction from a set of 2D images and corresponding camera poses. This repository demonstrates a concise end-to-end NeRF pipeline—from capturing or providing images, to training a NeRF model, to generating novel views and an optional 3D mesh.

## Overview

- **NeRF Model Implementation**: A PyTorch-based implementation following Mildenhall et al.’s original NeRF paper. The model learns a continuous volumetric representation that can be rendered from any viewpoint.
- **Data Capture & Preprocessing**: If you have a RealSense camera, you can run `scan.py` to capture images and compute poses automatically. Otherwise, a sample dataset is already included for quick testing.
- **Training & Rendering**: Train the NeRF using `pipeline.py`, then render novel views or a simple animation from arbitrary camera angles. All major hyperparameters and training settings are easily configurable.
- **Optional Mesh Extraction**: A utility to convert the trained NeRF density field into a polygon mesh (via Marching Cubes). Useful for visualization in standard 3D software.

## Quick Usage

1. **Install Dependencies**  
   ```bash
   git clone https://github.com/anandk1999/3D-Reconstruction-using-NeRFs.git
   cd 3D-Reconstruction-using-NeRFs
   pip install -r requirements.txt
   ```
   Make sure a GPU-compatible PyTorch is installed for faster training.

2. **(Optional) Capture Data**  
   Connect an Intel RealSense camera and run:
   ```bash
   python scan.py
   ```
   Press `c` to capture frames, then `q` to save.

3. **Convert & Train**  
   ```bash
   # Convert images & poses to training rays
   python pipeline.py convert

   # Train NeRF (defaults to 80k iterations)
   python pipeline.py train
   ```
   Checkpoints are saved periodically in `nerf_data/`.

4. **Render New Views**  
   ```bash
   python pipeline.py test
   ```
   Outputs appear in `novel_views/`.

5. **(Optional) Extract a Mesh**  
   ```bash
   python pipeline.py mesh
   python view.py mesh.obj
   ```
   Visualizes the mesh in an Open3D window.


## Project Structure (High-Level)

```
3D-Reconstruction-using-NeRFs/
├── pipeline.py      # Main training, testing, mesh extraction script
├── scan.py          # RealSense-based data capture
├── view.py          # Mesh viewer (Open3D)
├── requirements.txt # Project dependencies
├── nerf_data/       # Contains images, camera poses, and model checkpoints
└── ...              # Templates, utility scripts
└── training_data.pkl # Training data for NeRF model
└── testing_data.pkl # Testing Data
```

- **pipeline.py**: Central script for converting data, training the NeRF, rendering novel views, and optionally extracting a mesh.
- **scan.py**: Allows capturing images + camera poses using Intel RealSense (requires additional drivers).
- **view.py**: Loads and visualizes extracted meshes in 3D.


## References

- **NeRF Paper**: [*NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*](https://arxiv.org/abs/2003.08934), Mildenhall et al., ECCV 2020.
- **FreeNeRF**: *[FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization](https://arxiv.org/abs/2303.07418)*, Jiawei Yang, Marco Pavone, Yue Wang, CVPR 2023.
- **Intel RealSense & Open3D**: Helpful for camera-based data capture and 3D visualization.
- **Additional Variants**: Consider exploring Instant-NGP or other speed-optimized NeRF approaches once you understand this baseline.