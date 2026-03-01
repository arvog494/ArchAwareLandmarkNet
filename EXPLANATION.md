# 3D Teeth Pipeline Explanation

## Overview
This repo runs a full jaw pipeline: ToothGroupNetwork produces tooth instances,
DGCNNSeg4D assigns FDI labels with a 4D DGCNN and majority voting,
ArchAwareLandmarkNet predicts tooth landmarks, and a scanner path planner
generates collision-checked poses.

## Pipeline (end-to-end)
1. **Input and normalization**
   - Load a jaw mesh from `.obj` or `.stl`.
   - Rebase STL to the ToothGroupNetwork axis convention when needed
     (Y points to the back, Z up).
   - Center/scale points for model inputs.

2. **Tooth instance segmentation (ToothGroupNetwork)**
   - Run TGNet (`tgnet_fps` + `tgnet_bdl`) to produce per-vertex instance labels.
   - Align TGNet vertices to the current mesh and clean labels
     (small component removal, fill/smooth).

3. **FDI labeling (DGCNNSeg4D)**
   - Sample `n_points` from the mesh.
   - Compute arch index `t` per vertex from instances
     (fallback if instances are missing).
   - Normalize by full-mesh center/scale and build 4D features `(x,y,z,t)`.
   - Run DGCNNSeg4D, map class indices to FDI labels.
   - Propagate labels to all vertices with kNN + majority vote (`k=5`).

4. **Landmark prediction (ArchAwareLandmarkNet)**
   - For each tooth, sample jaw points and tooth points.
   - Predict 9 landmark points and a cusp-count class.

5. **Path detection and scanner placement**
   - Compute ordered keypoints (buccal, occlusal, lingual/palatal),
     project to mesh, and keep normals.
   - Fit a smooth spline through keypoints to generate a path.
   - Build an open-mouth box for collision checks.
   - Interpolate 6DoF orientations and place scanner poses with
     collision avoidance.

6. **Export**
   - Save normalized jaw, path points, 6DoF poses, and scanner geometry.

## Model architecture and algorithms

### ToothGroupNetwork (TGNet) - instance segmentation
Retrieve from a repo [GitHub repo from limhoyeon](https://github.com/limhoyeon/ToothGroupNetwork) and adpted for our code (modification and new functions)
- Code: `ToothGroupNetwork/models/modules/grouping_network_module.py` and
  `ToothGroupNetwork/models/modules/cbl_point_transformer`.
- Two-stage grouping network built on a Point Transformer encoder-decoder:
  - Each stage uses `PointTransformerSeg` with `TransitionDown`/`TransitionUp`
    blocks and multi-head outputs for class logits and center offsets.
  - Default `tgnet_fps` config (`ToothGroupNetwork/train_configs/tgnet_fps.py`):
    5 stages, planes `[32,64,128,256,512]`, blocks `[2,3,4,6,3]`,
    stride `[1,4,4,4,4]`, input features are 6D (xyz + normals).
  - `tgnet_bdl` uses a smaller 2-stage config for boundary-aware refinement.
- Stage 1 predicts coarse semantic labels and per-point center offsets.
  Offset-shifted points are clustered to estimate tooth centroids, then
  cropped (3072 points per tooth).
- Stage 2 refines boundaries using boundary-aware sampling.
- Final instance labels are mapped back to the full mesh by nearest neighbor.

### DGCNNSeg4D - FDI labeling with max voting
- Code: `visualization_dgcnn.ipynb` and `landmarks_prediction_v2.ipynb`.
- Input: 4D features `(x, y, z, t)`, where `t` is the arch index from TGNet
  instances (fallback if missing).
- Architecture:
  - Four EdgeConv blocks with kNN graph features (`k=20`).
  - Feature sizes: 64 -> 64 -> 128 -> 256.
  - Concatenate local features (512), apply global embedding
    (`conv5`, `emb_dims=1024`), then fuse local + global features.
  - Final per-point MLP with dropout and `conv8` to class logits.
- Output: class indices mapped to FDI labels.
- Max voting: predictions on sampled points are propagated to all vertices
  via kNN (`k=5`) and `np.bincount(...).argmax()` per vertex.

### ArchAwareLandmarkNet - per-tooth landmarks
- Code: `landmarks_prediction_v2.ipynb`.
- Inputs:
  - `jaw_pts` (global jaw point cloud).
  - `tooth_pts` (tooth-local point cloud).
  - `fdi` and `tooth_type` as 2D conditioning.
- Architecture:
  - GlobalPointNet: 1x1 conv MLP 64 -> 128 -> 256 with max + mean pooling.
  - ToothPointNet: 1x1 conv MLP 64 -> 128 -> 128 with max + mean pooling.
  - Conditioning MLP: 2 -> 32 -> cond_feat.
  - Fusion MLP: (global + tooth + cond) -> 512 -> 256 with dropout.
  - Heads: 9 landmark points (9x3) + cusp-count head (3 classes for 0, 2, 4).

### Scanner path creation algorithm
- Code: `landmarks_prediction_v2.ipynb`.
- Keypoint extraction:
  - Order teeth by FDI and group landmarks per tooth.
  - Occlusal midpoint from mesial/distal; add Z safety offset above cusps.
  - Lingual/palatal midpoint from inner point and cusps; buccal point from
    buccal landmark.
  - Project keypoints to the mesh and capture normals.
- Path generation:
  - Concatenate buccal + occlusal + lingual/palatal keypoints in arch order.
  - Fit a spline using arc-length parameterization (`splprep`/`splev`).
- 6DoF orientation:
  - Convert normals to rotation matrices (`normal_to_rotation_matrix`).
  - Interpolate rotations along the path (Slerp).
- Collision-aware placement:
  - Build an open-mouth box around the jaw.
  - Place the scanner so its tracking point follows the path.
  - Sweep yaw around global Z and incrementally lift +Z until collision-free
    (`CollisionManager` + FCL), keeping poses facing into the mouth.

## Metrics (from repo artifacts)
- ToothGroupNetwork:
  - Metrics reported in `ToothGroupNetwork/README.md` as TSA (IoU) and
    TIR (classification accuracy) figures. 95.99% IoU and 98.67% CLS.
  - Numeric values are not listed in the repo; only plots are provided.
- DGCNNSeg4D (FDI labels):
  - Tooth-only accuracy reported in `notebook_model.md` is ~0.916 on the test split before the max voting.
   - Metrics improve with max voting and almost match the ToothGroupeNetwork metrics
- ArchAwareLandmarkNet:
  - for landmarks without cusps : mean_dist 0.2139 unit (~cm) between true and predicted, for cusps : cusp_dist 0.5133 unit (~cm)
