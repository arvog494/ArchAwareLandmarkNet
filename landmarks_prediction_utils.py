"""Utilities extracted from landmarks_prediction_v2.ipynb."""

import os
import random
import json
import subprocess
import sys
import colorsys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from scipy.interpolate import interp1d, splprep, splev
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R, Slerp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from trimesh.collision import CollisionManager

import open3d as o3d
from settings import PROCESSED_DATA_PATH, DATA_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE = 0.5
RATIO = 1
CM_TO_UNIT = RATIO / SCALE
SCAN_OFFSET_CM = 1.5 * CM_TO_UNIT

SCAN_FIELD_DIM = {
    "L": 1.6 * CM_TO_UNIT,
    "W": 1.2 * CM_TO_UNIT,
    "H": 2.2 * CM_TO_UNIT,
}

OVERALL_FIELD_DIM = {
    "L": 28.1 * CM_TO_UNIT,
    "W": 3.3 * CM_TO_UNIT,
    "H": 4.6 * CM_TO_UNIT,
}

BASE_ORDER = ["InnerPoint", "FacialPoint", "Distal", "Mesial", "OuterPoint"]
MAX_CUSPS = 4

FDI_CKPT_PATH = Path("checkpoints") / "best_DGCNNSeg4D_no_gencive.pt"
LANDMARK_CKPT_PATH = Path("checkpoints") / "best_landmark_model.pt"

# Global holders populated by lazy loaders
model: nn.Module | None = None
idx_to_fdi = None
fdi_to_idx = None
_landmark_model_cache: nn.Module | None = None

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

def knn(x, k):
    # x : (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)   # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)       # (B, 1, N)
    dist = xx + xx.transpose(2, 1) + inner            # (B, N, N)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]    # plus proches
    return idx                                        # (B, N, k)

def get_graph_feature(x, k=20, idx=None):
    # x : (B, C, N) -> (B, 2C, N, k)
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)         # (B, N, k)
    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()      # (B, N, C)
    feature = x.view(B * N, C)[idx, :]
    feature = feature.view(B, N, k, C)      # (B, N, k, C)

    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3)   # (B, N, k, 2C)
    feature = feature.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    return feature

class DGCNNSeg4D(nn.Module):
    def __init__(self, num_classes: int, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = k

        # C_in = 4 -> 2C = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(512 + emb_dims, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        """
        x : (B, N, 4)  # (x,y,z,t)
        """
        B, N, _ = x.size()
        x = x.permute(0, 2, 1).contiguous()   # (B, 4, N)

        x1 = get_graph_feature(x, k=self.k)   # (B, 8, N, k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1)[0]                # (B, 64, N)

        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1)[0]                # (B, 64, N)

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1)[0]                # (B, 128, N)

        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1)[0]                # (B, 256, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x_global = self.conv5(x_cat)          # (B, emb_dims, N)
        x_max = x_global.max(dim=-1, keepdim=True)[0]    # (B, emb_dims, 1)
        x_global_expanded = x_max.repeat(1, 1, N)        # (B, emb_dims, N)

        x_final = torch.cat((x_cat, x_global_expanded), dim=1)  # (B, 512+emb_dims, N)

        x_final = self.conv6(x_final)         # (B, 512, N)
        x_final = self.conv7(x_final)         # (B, 256, N)
        x_final = self.dropout(x_final)
        x_final = self.conv8(x_final)         # (B, C, N)

        return x_final.permute(0, 2, 1).contiguous()  # (B, N, C)


def load_fdi_model(ckpt_path: Path | str = FDI_CKPT_PATH, device: torch.device = device):
    """Lazy loader for the FDI DGCNN checkpoint."""
    global model, idx_to_fdi, fdi_to_idx
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    fdi_to_idx = ckpt["fdi_to_idx"]
    idx_to_fdi = ckpt["idx_to_fdi"]
    num_classes = len(fdi_to_idx)
    model = DGCNNSeg4D(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_fdi_model():
    """Return the cached FDI model and index mappings, loading on first use."""
    if model is None or idx_to_fdi is None or fdi_to_idx is None:
        load_fdi_model()
    return model, idx_to_fdi, fdi_to_idx

def compute_arc_index_per_vertex(vertices: np.ndarray,
                                 instances: np.ndarray,
                                 jaw: str,
                                 axis: int = 0) -> np.ndarray:
    # m?me logique que dans JawPointCloudDatasetArc
    verts = vertices
    inst = np.asarray(instances)

    if inst.shape[0] != verts.shape[0]:
        print("Warning: instances/vertices length mismatch; using fallback arc-index.")
        return compute_arc_index_fallback(verts)

    inst_ids = sorted(int(i) for i in np.unique(inst) if i != 0)
    if len(inst_ids) == 0:
        return np.full((verts.shape[0],), 0.5, dtype=np.float32)

    centers = []
    for iid in inst_ids:
        mask = (inst == iid)
        if not np.any(mask):
            continue
        centers.append(verts[mask].mean(axis=0))
    centers = np.array(centers)
    inst_ids = np.array(inst_ids)

    order = np.argsort(centers[:, axis])
    inst_ids_ordered = inst_ids[order]

    n = len(inst_ids_ordered)
    if n > 1:
        t_values = np.linspace(0.0, 1.0, n, dtype=np.float32)
    else:
        t_values = np.array([0.5], dtype=np.float32)

    t_per_inst = {int(iid): float(t) for iid, t in zip(inst_ids_ordered, t_values)}

    t_full = np.zeros((verts.shape[0],), dtype=np.float32)
    for iid, t in t_per_inst.items():
        mask = (inst == iid)
        t_full[mask] = t

    t_full[inst == 0] = 0.5
    return t_full
    inst_ids = sorted(int(i) for i in np.unique(inst) if i != 0)
    if len(inst_ids) == 0:
        return np.full((verts.shape[0],), 0.5, dtype=np.float32)

    centers = []
    for iid in inst_ids:
        mask = (inst == iid)
        if not np.any(mask):
            continue
        centers.append(verts[mask].mean(axis=0))
    centers = np.array(centers)
    inst_ids = np.array(inst_ids)

    order = np.argsort(centers[:, axis])
    inst_ids_ordered = inst_ids[order]

    n = len(inst_ids_ordered)
    if n > 1:
        t_values = np.linspace(0.0, 1.0, n, dtype=np.float32)
    else:
        t_values = np.array([0.5], dtype=np.float32)

    t_per_inst = {int(iid): float(t) for iid, t in zip(inst_ids_ordered, t_values)}

    t_full = np.zeros((verts.shape[0],), dtype=np.float32)
    for iid, t in t_per_inst.items():
        mask = (inst == iid)
        t_full[mask] = t

    t_full[inst == 0] = 0.5
    return t_full

def predict_labels_for_sample(row, n_points=4096):
    mesh = trimesh.load_mesh(row["obj_path"], process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    labels   = np.asarray(row["labels"],   dtype=np.int64)
    instances = np.asarray(row["instances"], dtype=np.int64)
    jaw = row["jaw"]  # 'upper' ou 'lower'

    N = vertices.shape[0]
    if N >= n_points:
        idxs = np.random.choice(N, n_points, replace=False)
    else:
        idxs = np.random.choice(N, n_points, replace=True)

    pts = vertices[idxs]           # (N,3)
    gt_lbls = labels[idxs]         # (N,)
    inst_sampled = instances[idxs] # (N,)

    # t avant normalisation
    t_full = compute_arc_index_per_vertex(vertices, instances, jaw, axis=0)
    t = t_full[idxs][:, None]      # (N,1)

    # normalisation xyz comme à l'entraînement
    center = pts.mean(axis=0)
    pts_norm = pts - center
    scale = np.max(np.linalg.norm(pts_norm, axis=1))
    pts_norm = pts_norm / (scale + 1e-8)

    pts4 = np.concatenate([pts_norm, t], axis=1)  # (N,4)

    fdi_model, idx_map, _ = get_fdi_model()
    with torch.no_grad():
        x = torch.from_numpy(pts4[None, ...]).float().to(device)  # (1, N, 4)
        logits = fdi_model(x)                                     # (1, N, C)
        preds_idx = logits.argmax(dim=-1).cpu().numpy()[0]        # (N,)

    vec_map = np.vectorize(lambda i: idx_map[int(i)])
    preds_fdi = vec_map(preds_idx).astype(np.int64)

    return pts, preds_fdi, gt_lbls

def is_molar(fdi):
    """True si dent molaire (FDI 6-8)."""
    t = fdi % 10
    return t in (6, 7, 8)

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def _rotation_matrix_from_vectors(a, b):
    """Return R such that R @ a == b (approximately)."""
    a = _normalize(a)
    b = _normalize(b)
    if np.allclose(a, 0) or np.allclose(b, 0):
        return np.eye(3)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    # (anti)parallel
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a, axis))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    # Rodrigues
    k = v / s
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3) + K * s + (K @ K) * (1.0 - c)
    return R

def _apply_T_to_points(points, T):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]

def _transform_landmarks(ldmrks, T):
    out = []
    for lm in ldmrks:
        lm2 = lm.copy()
        if "coord" in lm2:
            p = np.asarray(lm2["coord"], dtype=np.float64).reshape(1, 3)
            lm2["coord"] = _apply_T_to_points(p, T).reshape(-1).tolist()
        out.append(lm2)
    return out

def compute_jaw_normalization_T(vertices, ldmrks):
    """
    T (4x4) such that:
    - Teeth point upward (+Z) (jaw on a table)
    - Buccal/facial side faces +Y (towards us), using FacialPoint vs InnerPoint when available
    - Translation: z_min->0, y_min->0, x centered (mid of bounds->0)
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("vertices must have shape (N,3)")

    # 1) Level: rotate thickness normal to +Z
    origin, normal_axis = get_jaw_frame(vertices, ldmrks)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    normal_axis = np.asarray(normal_axis, dtype=np.float64).reshape(3)
    R_level = _rotation_matrix_from_vectors(normal_axis, np.array([0.0, 0.0, 1.0]))

    v1 = (R_level @ (vertices - origin).T).T

    # 2) In-plane: X = major PCA axis in XY
    xy = v1[:, :2] - np.mean(v1[:, :2], axis=0)
    if xy.shape[0] >= 3:
        _, _, vh = np.linalg.svd(xy, full_matrices=False)
        x2d = vh[0]
    else:
        x2d = np.array([1.0, 0.0])

    x_axis = _normalize(np.array([x2d[0], x2d[1], 0.0]))
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))

    # 3) Choose sign so FacialPoint is +Y vs InnerPoint
    def _mean_class(class_name):
        pts = []
        for lm in ldmrks:
            if lm.get("class") != class_name or "coord" not in lm:
                continue
            p = np.asarray(lm["coord"], dtype=np.float64)
            pts.append(R_level @ (p - origin))
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    facial_mean = _mean_class("FacialPoint")
    inner_mean = _mean_class("InnerPoint")
    if facial_mean is not None and inner_mean is not None:
        facing_vec = facial_mean - inner_mean
        facing_vec[2] = 0.0
        if np.linalg.norm(facing_vec) > 1e-9 and float(np.dot(facing_vec, y_axis)) < 0.0:
            y_axis = -y_axis
            x_axis = -x_axis

    B = np.stack([x_axis, y_axis, z_axis], axis=1)
    R_inplane = B.T

    R = R_inplane @ R_level
    t0 = -R @ origin

    # 4) Translate to requested origin
    v_rot = (R @ vertices.T).T + t0
    x_center = 0.5 * (np.min(v_rot[:, 0]) + np.max(v_rot[:, 0]))
    y_min = np.min(v_rot[:, 1])
    z_min = np.min(v_rot[:, 2])
    shift = np.array([-x_center, -y_min, -z_min], dtype=np.float64)
    t = t0 + shift

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _build_T_from_R(vertices, R):
    """
    Build a translation that centers/grounds vertices after applying rotation R,
    following the same centering as compute_jaw_normalization_T:
      - x centered to 0
      - y min -> 0
      - z min -> 0
    """
    verts = np.asarray(vertices, dtype=np.float64)
    origin = verts.mean(axis=0)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t0 = -R @ origin
    v_rot = (R @ verts.T).T + t0
    x_center = 0.5 * (np.min(v_rot[:, 0]) + np.max(v_rot[:, 0]))
    y_min = np.min(v_rot[:, 1])
    z_min = np.min(v_rot[:, 2])
    shift = np.array([-x_center, -y_min, -z_min], dtype=np.float64)
    t = t0 + shift

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_jaw_frame(pts, ldmrks):
    """
    Estimate the jaw frame (Origin, Normal Axis).
    Normal Axis points 'Up' (towards cusps).
    """
    origin = np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    
    # v[:, 0] is smallest variance -> Normal (Thickness)
    normal_axis = v[:, 0]
    
    # Orient Normal towards Cusps
    cusps = [lm['coord'] for lm in ldmrks if lm.get('class') == 'Cusp']
    if cusps:
        cusp_mean = np.mean(cusps, axis=0)
        if np.dot(cusp_mean - origin, normal_axis) < 0:
            normal_axis = -normal_axis
            
    return origin, normal_axis

def get_landmarks_by_class(landmarks, class_name):
    """Return list of coordinates for a given landmark class."""
    return [np.array(lm['coord']) for lm in landmarks if lm.get('class') == class_name]

def group_landmarks_by_fdi(ldmrks):
    grouped = {}
    
    for lm in ldmrks:
        if 'fdi' in lm:
            grouped.setdefault(lm['fdi'], []).append(lm)
    return dict(grouped)

def order_teeth_fdi(fdi_list, upper=True):
    """
    Returns teeth ordered from left to right for upper or lower jaw.
    """
    fdi_list = sorted(set(fdi_list))

    if upper:
        right = [f for f in fdi_list if 21 <= f <= 28]
        left = [f for f in fdi_list if 11 <= f <= 18][::-1]
    else:
        right = [f for f in fdi_list if 31 <= f <= 38]
        left = [f for f in fdi_list if 41 <= f <= 48][::-1]

    return left + right

def compute_midpoint(pt1, pt2, *other_pts, w=0.5):

    if pt1 is None or pt2 is None:
        return None
    
    other_pts = [np.array(p) for p in other_pts if p is not None]
    
    soCMe_other = sum(other_pts) if other_pts else 0
    count_other = len(other_pts) if other_pts else 0

    return (pt1*w + pt2*(1-w) + soCMe_other) / (1 + count_other)

def get_occlusal_landmarks(landmarks):
    mesial = None
    distal = None
    cusps = []

    for lm in landmarks:
        if lm.get("class") == "Mesial":
            mesial = np.array(lm["coord"])
        elif lm.get("class") == "Distal":
            distal = np.array(lm["coord"])
        elif lm.get("class") == "Cusp":
            cusps.append(np.array(lm["coord"]))

    return mesial, distal, cusps

def get_lingualPalatal_landmarks(landmarks):

    mesial = None
    distal = None
    innerPoint = None
    cusps = []

    for lm in landmarks:
        if lm.get("class") == "Mesial":
            mesial = np.array(lm["coord"])
        elif lm.get("class") == "Distal":
            distal = np.array(lm["coord"])
        elif lm.get("class") == "InnerPoint":
            innerPoint = np.array(lm["coord"])
        elif lm.get("class") == "Cusp":
            cusps.append(np.array(lm["coord"]))

    return mesial, distal, innerPoint, cusps

def get_buccal_landmarks(landmarks):

    for lm in landmarks:
        if lm.get("class") == "FacialPoint":
            return np.array(lm["coord"])
        
    return None

def compute_offset_point(point, mesh):

    point = np.asarray(point, dtype=np.float64).reshape(1, 3)
    closest_pts, _ , face_ids = mesh.nearest.on_surface(point)
    nearest_vertex_id = mesh.kdtree.query(closest_pts)[1]
    normal = mesh.vertex_normals[nearest_vertex_id]
    safe_points = closest_pts + SCAN_OFFSET_CM * normal

    return safe_points.reshape(-1), closest_pts.reshape(-1), normal.reshape(-1)

def compute_highest_point(vertices):
    """
    Return the highest z coordinate from a list/array of vertices.
    vertices: array-like with shape (N, 3) or a single vertex (3,)
    Returns float z or None if vertices empty. Raises ValueError for invalid shapes.
    """
    verts = np.asarray(vertices, dtype=np.float64)

    # Empty input
    if verts.size == 0:
        return None

    # Single vertex as 1D array
    if verts.ndim == 1:
        if verts.shape[0] < 3:
            raise ValueError("Single vertex must have at least 3 coordinates (x,y,z)")
        return float(verts[2])

    # 2D array: (N, D) with D >= 3
    if verts.ndim == 2:
        if verts.shape[1] < 3:
            raise ValueError("Vertices must have at least 3 columns (x,y,z)")
        return float(np.max(verts[:, 2]))

    # Higher dimensions are not supported
    raise ValueError("Unsupported vertices array shape")

def compute_keypoints_ordered(ldmrks, mesh, vertices, labels, upper=True):

    fdi_list = list(set([lm.get('fdi') for lm in ldmrks.copy() if 'fdi' in lm]))

    fdi_ordered = order_teeth_fdi(fdi_list=fdi_list, upper=upper)
    ldmrks_by_fdi = group_landmarks_by_fdi(ldmrks)

    keypoints_occlusal = []
    keypoints_occlusal_projection = []
    keypoints_occlusal_projection_normals = []
    keypoints_lingualPalatal = []
    keypoints_lingualPalatal_projection = []
    keypoints_lingualPalatal_projection_normals = []
    keypoints_buccal = []
    keypoints_buccal_projection = []
    keypoints_buccal_projection_normals = []

    for fdi in fdi_ordered:

        lms = ldmrks_by_fdi.get(fdi, [])
        
        if not lms:
            keypoints_occlusal.append(None)
            keypoints_lingualPalatal.append(None)
            keypoints_buccal.append(None)
            continue

        # Occlusal midpoint
        mesial, distal, cusps = get_occlusal_landmarks(lms)

        if mesial is not None and distal is not None:
            occlusal_midpoint = compute_midpoint(mesial, distal)

            id_vertices_tooth = np.where(np.array(labels) == fdi)[0]
            vertices_tooth = vertices[id_vertices_tooth]
            z_offset = abs(compute_highest_point(vertices_tooth) - occlusal_midpoint[2])
            
            safe_points, proj_point, normal = compute_offset_point(occlusal_midpoint + np.array([0, 0, z_offset]), mesh)

            if cusps: safe_points = safe_points + np.array([0, 0, z_offset]) # to be above cusps
            keypoints_occlusal.append(safe_points)
            keypoints_occlusal_projection.append(proj_point)
            keypoints_occlusal_projection_normals.append(normal)
            
        # Lingual/Palatal midpoint
        mesial, distal, innerPoint, cusps = get_lingualPalatal_landmarks(lms)
        lingualPalatal_midpoint = None

        if innerPoint is not None:

            # cusp_arr = np.asarray(cusps)
            # if len(cusp_arr) >= 3:
            #     # Molar case (4 cusps): take two closest cusps -> midpoint -> blend twice with inner point
            #     cusp_dist = np.linalg.norm(cusp_arr - innerPoint, axis=1)
            #     closest_idxs = np.argsort(cusp_dist)[:2]
            #     c_mid = compute_midpoint(cusp_arr[closest_idxs[0]], cusp_arr[closest_idxs[1]])
            #     mid1 = compute_midpoint(c_mid, innerPoint)
            #     mid2 = compute_midpoint(mid1, innerPoint)
            #     mid3 = compute_midpoint(mid2, innerPoint)
            #     if mid3 is not None:
            #         lingualPalatal_midpoint = np.array([innerPoint[0], innerPoint[1], mid3[2]], dtype=np.float64)
            # elif len(cusp_arr) >= 1:
            #     # Premolar case (2 cusps): closest cusp -> midpoint chain with inner point
            #     cusp_dist = np.linalg.norm(cusp_arr - innerPoint, axis=1)
            #     closest_idx = int(np.argmin(cusp_dist))
            #     mid1 = compute_midpoint(cusp_arr[closest_idx], innerPoint)
            #     mid2 = compute_midpoint(mid1, innerPoint)
            #     mid3 = compute_midpoint(mid2, innerPoint)
            #     if mid3 is not None:
            #         lingualPalatal_midpoint = np.array([innerPoint[0], innerPoint[1], mid3[2]], dtype=np.float64)
            # elif mesial is not None or distal is not None:
            #     occlusal_midpoint = compute_midpoint(mesial, distal)
            #     lingualPalatal_midpoint = compute_midpoint(occlusal_midpoint, innerPoint)
            
            # if lingualPalatal_midpoint is not None:
            #     safe_points, proj_point, normal = compute_offset_point(lingualPalatal_midpoint, mesh)
            #     keypoints_lingualPalatal.append(safe_points)
            #     keypoints_lingualPalatal_projection.append(proj_point)
            #     keypoints_lingualPalatal_projection_normals.append(normal)
            safe_points, proj_point, normal = compute_offset_point(innerPoint, mesh)
            keypoints_lingualPalatal.append(safe_points)
            keypoints_lingualPalatal_projection.append(proj_point)
            keypoints_lingualPalatal_projection_normals.append(normal)
        
        # Buccal midpoint
        buccal_point = get_buccal_landmarks(lms)
        if buccal_point is not None:
            safe_points, proj_point, normal = compute_offset_point(buccal_point, mesh)
            keypoints_buccal.append(safe_points)
            keypoints_buccal_projection.append(proj_point)
            keypoints_buccal_projection_normals.append(normal)

    keypoints_occlusal.reverse()
    keypoints_occlusal_projection.reverse()
    keypoints_occlusal_projection_normals.reverse()
    keypoints = keypoints_buccal + keypoints_occlusal + keypoints_lingualPalatal
    projections = keypoints_buccal_projection + keypoints_occlusal_projection + keypoints_lingualPalatal_projection
    normals = keypoints_buccal_projection_normals + keypoints_occlusal_projection_normals + keypoints_lingualPalatal_projection_normals

    # Fallback: if no FDI/teeth keypoints were produced, use all landmark coords to create a minimal path
    if len(keypoints) == 0 or all(kp is None for kp in keypoints):
        coords = [np.asarray(lm["coord"], dtype=np.float64) for lm in ldmrks if "coord" in lm]
        if len(coords) >= 2:
            keypoints = coords
            projections = coords
            normals = [np.array([0.0, 0.0, 1.0], dtype=np.float64) for _ in coords]
        else:
            keypoints = []
            projections = []
            normals = []

    return keypoints, projections, normals

def generate_smooth_path(waypoints, num_points=300, smoothing=0.0):
    """
    Generate a smooth 3D path passing exactly through waypoints.
    
    Parameters
    ----------
    waypoints : (N,3) array
    num_points : number of samples on the path
    smoothing : 0 = exact interpolation
    
    Returns
    -------
    path : (num_points,3)
    """

    waypoints = np.asarray(waypoints)

    # Parameterize by cumulative arc length
    diffs = np.diff(waypoints, axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    u = np.insert(np.cumsum(dist), 0, 0)
    u /= u[-1]

    # Fit spline
    tck, _ = splprep(
        [waypoints[:,0], waypoints[:,1], waypoints[:,2]],
        u=u,
        s=smoothing,
        k=min(3, len(waypoints)-1)
    )

    # Sample spline
    u_fine = np.linspace(0, 1, num_points)
    x, y, z = splev(u_fine, tck)
    path = np.vstack((x, y, z)).T

    return path

def trimesh_path(path, color=[0, 0, 255, 255]):
    """
    Create a trimesh Path3D from Nx3 points
    """
    path = np.asarray(path)

    entities = [
        trimesh.path.entities.Line(
            points=np.arange(len(path))
        )
    ]

    return trimesh.path.Path3D(
        entities=entities,
        vertices=path,
        colors=[color]*len(entities)
    )

def waypoint_spheres(waypoints, radius=0.4, color=[255, 0, 0, 255]):
    """
    Represent waypoints as small spheres
    """
    spheres = []
    for p in waypoints:
        s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        s.apply_translation(p)
        s.visual.face_colors = color
        spheres.append(s)
    return spheres

def show_jaw_and_path_trimesh(jaw_mesh, waypoints, path):
    waypoints = np.asarray(waypoints)

    scene = trimesh.Scene()

    # Jaw
    jaw = jaw_mesh.copy()
    jaw.visual.face_colors = [200, 200, 200, 120]
    scene.add_geometry(jaw, node_name="jaw")

    # Path
    scene.add_geometry(
        trimesh_path(path),
        node_name="path"
    )

    # Waypoints
    for i, s in enumerate(waypoint_spheres(waypoints)):
        scene.add_geometry(s, node_name=f"wp_{i}")

    display(scene.show())

def normal_to_rotation_matrix(normal, up=np.array([0,0,1])):
    # Ensure normal is unit vector
    z_axis = normal / np.linalg.norm(normal)
    
    # Pick X-axis perpendicular to Z and some 'up' vector
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # up vector is aligned with normal, pick arbitrary
        up_alt = np.array([1,0,0])
        x_axis = np.cross(up_alt, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    # Y-axis
    y_axis = np.cross(z_axis, x_axis)
    
    # Rotation matrix: columns are x, y, z axes
    R = np.column_stack((x_axis, y_axis, z_axis))
    return R

def create_open_mouth_box(
    size=(120, 80, 60),      # width (X), depth (Y), height (Z)
    thickness=2.0,
    opening_axis="+Y"
):
    """
    Create a box open on one side (mouth opening).

    Returns a trimesh.Trimesh
    """
    W, D, H = size
    t = thickness

    parts = []

    # Bottom
    bottom = trimesh.creation.box(extents=(W, D, t))
    bottom.apply_translation([0, 0, -H/2])
    parts.append(bottom)

    # Top
    top = trimesh.creation.box(extents=(W, D, t))
    top.apply_translation([0, 0, H/2])
    parts.append(top)

    # Left
    left = trimesh.creation.box(extents=(t, D, H))
    left.apply_translation([-W/2, 0, 0])
    parts.append(left)

    # Right
    right = trimesh.creation.box(extents=(t, D, H))
    right.apply_translation([W/2, 0, 0])
    parts.append(right)

    # Back (throat side)
    back = trimesh.creation.box(extents=(W, t, H))
    back.apply_translation([0, -D/2, 0])
    parts.append(back)

    box = trimesh.util.concatenate(parts)
    box.visual.face_colors = [180, 180, 180, 100]
    return box

def create_rectangular_frustum(Wf, Hf, Wb, Hb, L):
    """
    Rectangular frustum along +X
    Front face at x=0, back face at x=L
    Z is up, Y is left
    """

    # Front face (scan field side)
    v0 = [0, -Wf/2, 0]
    v1 = [0,  Wf/2, 0]
    v2 = [0,  Wf/2, Hf]
    v3 = [0, -Wf/2, Hf]

    # Back face (body side)
    v4 = [L, -Wb/2, 0]
    v5 = [L,  Wb/2, 0]
    v6 = [L,  Wb/2, Hb]
    v7 = [L, -Wb/2, Hb]

    vertices = np.array([v0,v1,v2,v3,v4,v5,v6,v7])

    faces = np.array([
        # Front
        [0,1,2], [0,2,3],
        # Back
        [4,6,5], [4,7,6],
        # Bottom
        [0,4,5], [0,5,1],
        # Top
        [3,2,6], [3,6,7],
        # Left
        [0,3,7], [0,7,4],
        # Right
        [1,5,6], [1,6,2],
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

def create_scanner_mesh(scan_dim, overall_dim):
    """
    Origin = center of bottom face of scan field.
    """

    # --- Dimensions ---
    Ls = scan_dim["L"]
    Ws = scan_dim["H"]
    Hs = scan_dim["W"]

    Lb = overall_dim["L"]
    Wb = overall_dim["H"]
    Hb = overall_dim["W"]

    # --- Scan field ---
    scan = trimesh.creation.box(extents=[Ls, Ws, Hs])
    scan.apply_translation([Ls/2, 0, Hs/2])

    # --- Connector ---
    connector_length = Lb
    connector = create_rectangular_frustum(
        Wf=Ws, Hf=Hs,
        Wb=Wb, Hb=Hb,
        L=connector_length
    )
    connector.fix_normals()
    connector.apply_translation([Ls, 0, 0])

    scanner = trimesh.util.concatenate([scan, connector])
    scanner.visual.face_colors = [255, 0, 0, 120]

    return scanner

def scanner_tracking_point_local(scan_dim):
    """
    Local coordinates of the point that follows the path.
    Bottom-center of scan field.
    """
    L, W, H = scan_dim["L"], scan_dim["W"], scan_dim["H"]

    return np.array([
        L/2,        # front of scan field
        0.0,        # centered in width
        0.0
    ])

def tracking_point_world(path_point, R, local_point):
    return path_point + R @ local_point

def show_scene_with_tracking_point(
    jaw_mesh,
    path,
    scanner_mesh,
    scan_dim,
    R_list,
    index=0
):
    scene = trimesh.Scene()

    # Jaw
    jaw_mesh.visual.face_colors = [200, 200, 200, 80]
    scene.add_geometry(jaw_mesh, node_name="jaw")

    # Path
    path_geom = trimesh.load_path(path)
    scene.add_geometry(path_geom, node_name="path")

    # Scanner pose: we want the tracking point to coincide with path[index]
    local_p = scanner_tracking_point_local(scan_dim)

    scanner = scanner_mesh.copy()
    scanner.apply_translation(-local_p)  # put tracking point at local origin

    T = np.eye(4)
    T[:3, :3] = R_list[index]
    T[:3, 3] = path[index]  # now this IS the tracking point in world coords

    scanner.apply_transform(T)
    scene.add_geometry(scanner, node_name="scanner")

    # Visualize tracking point (must sit exactly on the path point)
    tracking_sphere = trimesh.creation.icosphere(radius=0.8)
    tracking_sphere.apply_translation(path[index])
    tracking_sphere.visual.face_colors = [0, 0, 255, 255]
    scene.add_geometry(tracking_sphere, node_name="tracking_point")

    display(scene.show(smooth=False))

def place_scanner(scanner_mesh, point, rotation_matrix, scan_dim=SCAN_FIELD_DIM):
    """
    Place scanner mesh so that the scanner tracking point sits on `point`.
    """
    placed = scanner_mesh.copy()

    # Move tracking point to local origin first
    local_p = scanner_tracking_point_local(scan_dim)
    placed.apply_translation(-local_p)

    # Apply world pose
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = point
    placed.apply_transform(T)

    return placed

def _ensure_rot_matrix(rot):
    """
    Accepts either:
      - scipy Rotation
      - (3,3) rotation matrix
    Returns (3,3) numpy rotation matrix.
    """
    if isinstance(rot, R):
        return rot.as_matrix()
    rot = np.asarray(rot)
    if rot.shape == (3, 3):
        return rot
    raise ValueError(f"Unsupported rotation type/shape: {type(rot)} / {getattr(rot, 'shape', None)}")

def find_collision_free_pose_interp_rot(
    base_point,
    base_rot,
    scanner_mesh,
    obstacles_manager: CollisionManager,
    *,
    z_rot_step_deg: float = 10.0,
    z_lift_step: float = 0.5,
    max_z_lifts: int = 40,
    debug: bool = True,
    idx: int | None = None,
    scan_dim=SCAN_FIELD_DIM,
):
    """
    Tries:
      1) pose at base_point with base_rot
      2) if collision: yaw sweep around GLOBAL Z
      3) if still collision after full turn: lift +Z and retry

    Returns:
      (pt_ok, R_ok, yaw_deg, z_lift)
    or:
      (None, None, None, None)
    """
    base_R = _ensure_rot_matrix(base_rot)

    # Pre-shift scanner so its tracking point becomes the local origin
    local_p = scanner_tracking_point_local(scan_dim)
    scanner_local = scanner_mesh.copy()
    scanner_local.apply_translation(-local_p)

    yaw_steps = max(1, int(round(360.0 / z_rot_step_deg)))
    yaw_angles = np.linspace(0.0, 360.0, yaw_steps, endpoint=False)

    for lift_i in range(max_z_lifts + 1):
        z_lift = lift_i * z_lift_step
        local_z_world = base_R[:, 2]
        pt = np.array(base_point, dtype=float) + z_lift * local_z_world

        if debug:
            tag = f"i={idx} " if idx is not None else ""
            print(f"[TRY {tag}] base={np.asarray(base_point)} lift={z_lift:.2f}")

        for yaw_deg in yaw_angles:
            yaw_R = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
            R_try = base_R @ yaw_R   # yaw around LOCAL Z of base_R

            T = np.eye(4)
            T[:3, :3] = R_try
            T[:3, 3] = pt

            placed = scanner_local.copy()
            placed.apply_transform(T)

            collided = obstacles_manager.in_collision_single(placed)

            scanner_forward_local = np.array([1.0, 0.0, 0.0])
            scanner_forward_world = R_try @ scanner_forward_local

            mouth_dir_world = np.array([0.0, -1.0, 0.0])  # negative Y points into mouth
            mouth_dir_world /= np.linalg.norm(mouth_dir_world)

            # small debug: print only on success
            if debug and not collided and np.dot(scanner_forward_world, mouth_dir_world) <= 0.0:
                tag = f"i={idx} " if idx is not None else ""
                print(f"  [{tag}lift={z_lift:.2f}] yaw={yaw_deg:6.1f}° collided={collided}")

            if not collided and np.dot(scanner_forward_world, mouth_dir_world) <= 0.0:
                if debug:
                    tag = f"i={idx} " if idx is not None else ""
                    print(f"[OK  {tag}] pt={pt} yaw={yaw_deg:.1f}° lift={z_lift:.2f}")
                return pt, R_try, yaw_deg, z_lift

        if debug:
            tag = f"i={idx} " if idx is not None else ""
            print(f"[NO_YAW {tag}] lift={z_lift:.2f} -> lifting +Z")

    if debug:
        tag = f"i={idx} " if idx is not None else ""
        print(f"[FAIL {tag}] No collision-free pose up to lift={max_z_lifts * z_lift_step:.2f}")

    return None, None, None, None

def place_scanner_along_path_interp_rots(
    path_pts,
    interp_rots,
    scanner_mesh,
    obstacles,
    *,
    z_rot_step_deg: float = 10.0,
    z_lift_step: float = 0.5,
    max_z_lifts: int = 40,
    debug: bool = True,
    scan_dim=SCAN_FIELD_DIM,
):
    if len(path_pts) != len(interp_rots):
        raise ValueError(f"len(path_pts)={len(path_pts)} != len(interp_rots)={len(interp_rots)}")

    manager = CollisionManager()
    for j, obs in enumerate(obstacles):
        manager.add_object(f"obs_{j}", obs)

    points_out = []
    R_out = []
    kept_idx = []

    for i, (pt, rot) in enumerate(zip(path_pts, interp_rots)):
        pt_ok, R_ok, yaw_deg, z_lift = find_collision_free_pose_interp_rot(
            base_point=pt,
            base_rot=rot,
            scanner_mesh=scanner_mesh,
            obstacles_manager=manager,
            z_rot_step_deg=z_rot_step_deg,
            z_lift_step=z_lift_step,
            max_z_lifts=max_z_lifts,
            debug=debug,
            idx=i,
            scan_dim=scan_dim,
        )

        if pt_ok is None:
            # Retry with Z-flipped rotation before skipping
            try:
                base_R = _ensure_rot_matrix(rot)
                R_flip_z = base_R @ np.diag([1.0, 1.0, -1.0])
                pt_ok, R_ok, yaw_deg, z_lift = find_collision_free_pose_interp_rot(
                    base_point=pt,
                    base_rot=R_flip_z,
                    scanner_mesh=scanner_mesh,
                    obstacles_manager=manager,
                    z_rot_step_deg=z_rot_step_deg,
                    z_lift_step=z_lift_step,
                    max_z_lifts=max_z_lifts,
                    debug=debug,
                    idx=i,
                    scan_dim=scan_dim,
                )
            except Exception:
                pt_ok = None
                R_ok = None

        if pt_ok is None:
            print(f"[SKIP] {i}")
            continue

        points_out.append(pt_ok)
        R_out.append(R_ok)
        kept_idx.append(i)

    return np.array(points_out), R_out, kept_idx

def show_all_scanner_poses(
    jaw_mesh,
    path_pts,
    R_list,
    scanner_mesh,
    scan_dim,
    *,
    stride=5,
    merge=True,
    scanner_color=(255, 0, 0, 50),
    jaw_color=(200, 200, 200, 80),
    show_path=True,
    show_tracking_points=False,
    tracking_radius=0.6,
):
    """
    Visualize many scanner poses in a single trimesh.Scene.

    Assumptions:
      - `path_pts[i]` is the desired WORLD position of the tracking point.
      - `R_list[i]` is a (3,3) rotation matrix in WORLD frame.
      - `scanner_mesh` is defined in its local frame.
      - `scanner_tracking_point_local(scan_dim)` exists and returns the local tracking point.

    Notes:
      - Showing all poses can be heavy; use `stride` to subsample (e.g. 5, 10).
      - If `merge=True`, scanners are concatenated into one mesh for faster rendering.
    """
    if len(path_pts) != len(R_list):
        raise ValueError(f"len(path_pts)={len(path_pts)} != len(R_list)={len(R_list)}")

    scene = trimesh.Scene()

    # Jaw
    jaw = jaw_mesh.copy()
    jaw.visual.face_colors = np.array(jaw_color, dtype=np.uint8)
    scene.add_geometry(jaw, node_name="jaw")

    # Optional path polyline
    if show_path:
        try:
            path_geom = trimesh.load_path(np.asarray(path_pts))
            scene.add_geometry(path_geom, node_name="path")
        except Exception:
            pass

    local_tp = scanner_tracking_point_local(scan_dim)

    placed_scanners = []
    idxs = list(range(0, len(path_pts), max(1, int(stride))))

    for k, i in enumerate(idxs):
        pt = np.asarray(path_pts[i], dtype=float)
        Rmat = np.asarray(R_list[i], dtype=float)

        # place scanner so that tracking point is on pt
        sc = scanner_mesh.copy()
        sc.apply_translation(-local_tp)

        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = pt
        sc.apply_transform(T)

        sc.visual.face_colors = np.array(scanner_color, dtype=np.uint8)

        if merge:
            placed_scanners.append(sc)
        else:
            scene.add_geometry(sc, node_name=f"scanner_{i}")

        if show_tracking_points:
            sph = trimesh.creation.icosphere(radius=tracking_radius)
            sph.apply_translation(pt)
            sph.visual.face_colors = np.array([0, 0, 255, 255], dtype=np.uint8)
            scene.add_geometry(sph, node_name=f"tp_{i}")

    if merge and placed_scanners:
        merged = trimesh.util.concatenate(placed_scanners)
        merged.visual.face_colors = np.array(scanner_color, dtype=np.uint8)
        scene.add_geometry(merged, node_name="scanners_merged")

    display(scene.show(smooth=False))
    return scene

def assign_fdi_to_landmarks(
    ldmrks,
    vertices,
    vertex_labels,
    k=5
):
    """
    Assign FDI label to each landmark using nearest vertices.
    """

    # force numpy array
    vertex_labels = np.asarray(vertex_labels, dtype=np.int8)

    tree = cKDTree(vertices)

    out = []

    for lm in ldmrks:

        if 'coord' not in lm:
            continue

        # Normalize landmark
        coord = np.asarray(lm['coord'], dtype=np.float32)

        # Query KDTree
        _, idxs = tree.query(coord, k=k)

        # Handle k=1 case
        if np.isscalar(idxs):
            idxs = np.array([idxs], dtype=np.float32)

        neighbor_labels = vertex_labels[idxs]

        neighbor_labels = neighbor_labels[neighbor_labels > 0]
        if len(neighbor_labels) == 0:
            continue
            
        # Majority vote
        fdi = np.bincount(neighbor_labels).argmax()

        lm_clean = lm.copy()
        lm_clean['coord'] = coord.tolist()
        lm_clean['fdi'] = int(fdi)

        out.append(lm_clean)

    return out

class JawLdmrksDataset(Dataset):
    """
    Dataset for jaw meshes with landmarks.
    """

    def __init__(self, df, split="train", preload=True):

        self.df = df[df["split"] == split].reset_index(drop=True)

        if preload:
            self.vertices_list = []
            self.ldmrks_list = []
            self.mesh_list = []
            self.labels_list = []  # Store FDI labels
            
            for _, row in self.df.iterrows():
                path = row["obj_path"].replace("\\", "/")
                mesh = trimesh.load_mesh(Path(path), process=False)
                self.mesh_list.append(mesh)
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                self.vertices_list.append(vertices)
                labels = row["labels"].copy()
                self.labels_list.append(labels)
                self.ldmrks_list.append(assign_fdi_to_landmarks(row["ldmrks"], vertices, labels))
        else:
            self.vertices_list = []
            self.ldmrks_list = []
            self.mesh_list = []
            self.labels_list = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.vertices_list and self.ldmrks_list:

            vertices = self.vertices_list[idx].copy()
            ldmrks = self.ldmrks_list[idx].copy()
            mesh = self.mesh_list[idx]
            labels = self.labels_list[idx].copy()

        else:

            row = self.df.iloc[idx]
            mesh = trimesh.load_mesh(row["obj_path"], process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            labels = row["labels"].copy()
            ldmrks = assign_fdi_to_landmarks(row["ldmrks"], vertices, labels)

        return mesh, vertices, labels, ldmrks

def tooth_type_from_fdi(fdi):
    t = fdi % 10
    if t in (1, 2):
        return 0  # incisor
    if t == 3:
        return 1  # canine
    if t in (4, 5):
        return 2  # premolar
    return 3      # molar

def expected_cusp_count(fdi):
    ttype = tooth_type_from_fdi(fdi)
    if ttype == 2:
        return 2
    if ttype == 3:
        return 4
    return 0

def pack_landmarks_by_fdi(ldmrks, fdi):
    pts = {}
    cusps = []
    ttype = tooth_type_from_fdi(fdi)
    allow_cusps = ttype >= 2
    for lm in ldmrks:
        if lm.get("fdi") != fdi or "coord" not in lm or "class" not in lm:
            continue
        cls = lm["class"]
        p = np.array(lm["coord"], dtype=np.float32)
        if cls.startswith("Cusp"):
            if allow_cusps:
                cusps.append(p)
            continue
        pts[cls] = p

    out = []
    mask = []
    for name in BASE_ORDER:
        if name in pts:
            out.append(pts[name]); mask.append(1.0)
        else:
            out.append(np.zeros(3, dtype=np.float32)); mask.append(0.0)

    cusp_array = cusps[:MAX_CUSPS]
    while len(cusp_array) < MAX_CUSPS:
        cusp_array.append(np.zeros(3, dtype=np.float32))

    out.extend(cusp_array)
    mask.extend([1.0 if i < len(cusps) else 0.0 for i in range(MAX_CUSPS)])

    out = np.stack(out, axis=0)
    mask = np.array(mask, dtype=np.float32)
    return out, mask, cusps

def _sample_points(pts, n_points):
    if n_points is None:
        return pts
    n = pts.shape[0]
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        idx = np.random.choice(n, n_points, replace=True)
    return pts[idx]

def _apply_ref_axis_fix(T_jaw, ref_R=None):
    if ref_R is None:
        return T_jaw
    ref_R = np.asarray(ref_R, dtype=np.float64)
    if ref_R.shape != (3, 3):
        return T_jaw
    R = T_jaw[:3, :3]
    t = T_jaw[:3, 3]
    if float(np.dot(R[1], ref_R[1])) < 0.0:
        R_flip = np.diag([-1.0, -1.0, 1.0])
        R = R_flip @ R
        t = R_flip @ t
        T_jaw = np.eye(4, dtype=np.float64)
        T_jaw[:3, :3] = R
        T_jaw[:3, 3] = t
    return T_jaw


def load_reference_rotation(ref_mesh_path, *, df_lookup=None):
    """
    Compute reference jaw rotation from a known mesh (and optional landmarks from df_lookup).
    Returns a (3,3) rotation matrix or None on failure.
    """
    try:
        ref_mesh = load_mesh_any(ref_mesh_path)
        ref_ldmrks = []
        if df_lookup is not None:
            row = _find_row_by_mesh_path(df_lookup, ref_mesh_path)
            if row is not None and "ldmrks" in row:
                ref_ldmrks = row["ldmrks"]
        T_ref = compute_jaw_normalization_T(np.asarray(ref_mesh.vertices, dtype=np.float64), ref_ldmrks)
        return T_ref[:3, :3]
    except Exception as exc:
        print(f"Warning: could not compute reference rotation from {ref_mesh_path}: {exc}")
        return None

def apply_jaw_normalization(vertices, ldmrks, ref_R=None, offset=None):
    T_jaw = compute_jaw_normalization_T(vertices, ldmrks)
    # If a reference rotation is provided, align to it (translation recomputed)
    if ref_R is not None:
        T_jaw = _build_T_from_R(vertices, ref_R)
    else:
        T_jaw = _apply_ref_axis_fix(T_jaw, ref_R=ref_R)
    if offset is not None:
        off = np.asarray(offset, dtype=np.float64).reshape(3)
        T_jaw[:3, 3] += off
    v_norm = _apply_T_to_points(vertices, T_jaw)
    l_norm = _transform_landmarks(ldmrks, T_jaw)
    return v_norm, l_norm, T_jaw

def random_z_rotation_T(max_deg=10.0, scale_range=0.02):
    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    R = R * scale
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T

def apply_random_augmentation(vertices, ldmrks, max_deg=10.0, scale_range=0.02, p=0.9):
    if np.random.rand() > p:
        return vertices, ldmrks
    T = random_z_rotation_T(max_deg=max_deg, scale_range=scale_range)
    v_aug = _apply_T_to_points(vertices, T)
    l_aug = _transform_landmarks(ldmrks, T)
    return v_aug, l_aug

def compute_tooth_norm_params(tooth_vertices):
    tooth_vertices = np.asarray(tooth_vertices, dtype=np.float32)
    center = np.mean(tooth_vertices, axis=0)
    scale = np.max(np.linalg.norm(tooth_vertices - center, axis=1)) + 1e-8
    return center, float(scale)

def apply_tooth_norm(points, center, scale):
    pts = np.asarray(points, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    return (pts - c) / np.float32(scale)

def invert_tooth_norm(points, center, scale):
    pts = np.asarray(points, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    return pts * np.float32(scale) + c

class ToothWithArchDataset(Dataset):
    def __init__(self, df, split="train", jaw_points=8192, tooth_points=2048, augment=True, local_norm=True):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.jaw_points = jaw_points
        self.tooth_points = tooth_points
        self.split = split
        self.augment = augment
        self.local_norm = local_norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mesh = trimesh.load_mesh(row["obj_path"], process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        labels = np.asarray(row["labels"], dtype=np.int64)

        # Assign landmark FDI using true vertex labels
        ldmrks = assign_fdi_to_landmarks(row["ldmrks"], vertices, labels)

        # Jaw normalization using your pipeline
        vertices, ldmrks, _ = apply_jaw_normalization(vertices, ldmrks)

        if self.augment and self.split == "train":
            vertices, ldmrks = apply_random_augmentation(vertices, ldmrks)

        jaw_pts = _sample_points(vertices, self.jaw_points)

        tooth_ids = [int(f) for f in np.unique(labels) if f > 0]
        samples = []
        for fdi in tooth_ids:
            mask = labels == fdi
            if mask.sum() < 32:
                continue

            tooth_vertices = vertices[mask]
            tooth_pts = _sample_points(tooth_vertices, self.tooth_points)
            target, target_mask, cusps = pack_landmarks_by_fdi(ldmrks, fdi)

            base_mask = target_mask[:len(BASE_ORDER)]
            if base_mask.sum() < len(BASE_ORDER):
                continue

            cusp_count = expected_cusp_count(fdi)
            if cusp_count > 0 and len(cusps) < cusp_count:
                continue

            if self.local_norm:
                center, scale = compute_tooth_norm_params(tooth_vertices)
                jaw_pts_local = apply_tooth_norm(jaw_pts, center, scale)
                tooth_pts_local = apply_tooth_norm(tooth_pts, center, scale)
                target_local = apply_tooth_norm(target, center, scale)
                cusps_local = [apply_tooth_norm(c, center, scale) for c in cusps]

            samples.append({
                "jaw_pts": jaw_pts_local if self.local_norm else jaw_pts,
                "tooth_pts": tooth_pts_local if self.local_norm else tooth_pts,
                "fdi": int(fdi),
                "tooth_type": tooth_type_from_fdi(fdi),
                "target": target_local if self.local_norm else target,
                "mask": target_mask,
                "cusps": cusps_local if self.local_norm else cusps,
                "cusp_count": cusp_count,
            })

        return samples

class GlobalPointNet(nn.Module):
    def __init__(self, in_ch=3, feat=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, feat, 1), nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        f = self.mlp(x)
        g_max = torch.max(f, dim=-1)[0]
        g_mean = torch.mean(f, dim=-1)
        return torch.cat([g_max, g_mean], dim=1)

class ToothPointNet(nn.Module):
    def __init__(self, in_ch=3, feat=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, 64, 1), nn.ReLU(),
            nn.Conv1d(64, feat, 1), nn.ReLU(),
            nn.Conv1d(feat, feat, 1), nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        f = self.mlp(x)
        g_max = torch.max(f, dim=-1)[0]
        g_mean = torch.mean(f, dim=-1)
        return torch.cat([g_max, g_mean], dim=1)

class ArchAwareLandmarkNet(nn.Module):
    def __init__(self, global_feat=256, tooth_feat=128, cond_feat=32, out_points=9):
        super().__init__()
        self.global_net = GlobalPointNet(3, global_feat)
        self.tooth_net = ToothPointNet(3, tooth_feat)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, cond_feat), nn.ReLU()
        )
        in_dim = global_feat * 2 + tooth_feat * 2 + cond_feat
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.head = nn.Linear(256, out_points * 3)
        self.cusp_count_head = nn.Linear(256, 3)  # 0,2,4

    def forward(self, jaw_pts, tooth_pts, fdi, ttype):
        g = self.global_net(jaw_pts)
        t = self.tooth_net(tooth_pts)
        cond = torch.stack([fdi.float(), ttype.float()], dim=1)
        cond = self.cond_mlp(cond)
        x = torch.cat([g, t, cond], dim=1)
        x = self.fusion(x)
        out = self.head(x).view(-1, 9, 3)
        cc = self.cusp_count_head(x)
        return out, cc


def load_landmark_model(ckpt_path: Path | str = LANDMARK_CKPT_PATH, device: torch.device = device):
    """Load the ArchAwareLandmarkNet checkpoint and return a ready-to-use model."""
    lm_model = ArchAwareLandmarkNet().to(device)
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    lm_model.load_state_dict(ckpt["model_state"])
    lm_model.eval()
    return lm_model


def get_landmark_model():
    """Return a cached landmark model (loads the checkpoint on first use)."""
    global _landmark_model_cache
    if _landmark_model_cache is None:
        _landmark_model_cache = load_landmark_model()
    return _landmark_model_cache


def cusp_count_to_class(cusp_count):
    if cusp_count == 2:
        return 1
    if cusp_count == 4:
        return 2
    return 0

def landmark_loss(pred, target, mask, beta=0.01):
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).sum(dim=-1)
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-6)

def surface_loss(pred, tooth_pts, mask):
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    pred_f = pred.float()
    tooth_f = tooth_pts.float()
    dist = torch.cdist(pred_f, tooth_f, p=2)
    min_dist = dist.min(dim=-1).values
    min_dist = min_dist * mask
    return min_dist.sum() / (mask.sum() + 1e-6)

def cusp_set_loss(pred_cusps, gt_cusps_list):
    losses = []
    for b in range(pred_cusps.shape[0]):
        gt = gt_cusps_list[b]
        if gt is None or len(gt) == 0:
            losses.append(pred_cusps[b].abs().sum() * 0.0)
            continue
        if torch.is_tensor(gt):
            g = gt
        elif isinstance(gt, (list, tuple)) and len(gt) > 0 and torch.is_tensor(gt[0]):
            g = torch.stack(gt, dim=0)
        else:
            g = torch.as_tensor(gt)
        g = g.to(device=pred_cusps.device, dtype=pred_cusps.dtype)
        if g.ndim == 1:
            g = g.view(1, -1)
        if g.ndim > 2:
            g = g.reshape(-1, g.shape[-1])
        if g.shape[-1] != 3:
            raise ValueError(f"Expected cusp coords with 3 values, got shape {tuple(g.shape)}")
        p = pred_cusps[b].float()
        g = g.float()
        cost = torch.cdist(p, g, p=1)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        losses.append(cost[row, col].sum() / max(1, g.shape[0]))
    return torch.stack(losses).mean()

def init_metrics():
    return {
        "loss_sum": 0.0,
        "loss_count": 0,
        "dist_sum": 0.0,
        "dist_count": 0.0,
        "base_dist_sum": 0.0,
        "base_count": 0.0,
        "cusp_dist_sum": 0.0,
        "cusp_count": 0.0,
        "cc_correct": 0.0,
        "cc_total": 0.0,
    }

def update_metrics(metrics, pred, target, mask, cc_logits=None, cc_target=None):
    with torch.no_grad():
        dist = torch.linalg.norm(pred - target, dim=-1)
        if mask.ndim == 1:
            mask = mask.view(1, -1)
        metrics["dist_sum"] += (dist * mask).sum().item()
        metrics["dist_count"] += mask.sum().item()

        base_mask = mask[:, :5]
        cusp_mask = mask[:, 5:9]
        metrics["base_dist_sum"] += (dist[:, :5] * base_mask).sum().item()
        metrics["base_count"] += base_mask.sum().item()
        metrics["cusp_dist_sum"] += (dist[:, 5:9] * cusp_mask).sum().item()
        metrics["cusp_count"] += cusp_mask.sum().item()

        if cc_logits is not None and cc_target is not None:
            cc_pred = torch.argmax(cc_logits, dim=-1)
            metrics["cc_correct"] += (cc_pred == cc_target).sum().item()
            metrics["cc_total"] += cc_target.numel()

def summarize_metrics(metrics):
    out = {}
    out["loss"] = metrics["loss_sum"] / max(1, metrics["loss_count"])
    out["mean_dist"] = metrics["dist_sum"] / max(1, metrics["dist_count"])
    out["base_mean"] = metrics["base_dist_sum"] / max(1, metrics["base_count"])
    if metrics["cusp_count"] > 0:
        out["cusp_mean"] = metrics["cusp_dist_sum"] / metrics["cusp_count"]
    else:
        out["cusp_mean"] = float("nan")
    out["cc_acc"] = metrics["cc_correct"] / max(1, metrics["cc_total"])
    return out

def run_epoch(loader, model, optimizer=None, scaler=None, train=False, use_amp=False, desc=None):
    metrics = init_metrics()
    if train:
        model.train()
    else:
        model.eval()

    for batch in tqdm(loader, desc=desc):
        if len(batch) == 0:
            continue

        loss_total = 0.0
        for sample in batch:
            jaw = sample["jaw_pts"]
            if not torch.is_tensor(jaw):
                jaw = torch.from_numpy(jaw)
            jaw = jaw.float().to(device, non_blocking=pin_memory)

            tooth = sample["tooth_pts"]
            if not torch.is_tensor(tooth):
                tooth = torch.from_numpy(tooth)
            tooth = tooth.float().to(device, non_blocking=pin_memory)

            fdi = sample["fdi"]
            if not torch.is_tensor(fdi):
                fdi = torch.tensor([fdi], dtype=torch.float32)
            fdi = fdi.float().to(device, non_blocking=pin_memory)

            ttype = sample["tooth_type"]
            if not torch.is_tensor(ttype):
                ttype = torch.tensor([ttype], dtype=torch.float32)
            ttype = ttype.float().to(device, non_blocking=pin_memory)

            target = torch.as_tensor(sample["target"]).float().to(device, non_blocking=pin_memory)
            mask = torch.as_tensor(sample["mask"]).float().to(device, non_blocking=pin_memory)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred, cc_logits = model(jaw, tooth, fdi, ttype)

                    loss_base = landmark_loss(pred, target, mask)
                    pred_cusps = pred[:, 5:9, :]
                    loss_cusp = cusp_set_loss(pred_cusps, [sample["cusps"]])
                    loss_surface = surface_loss(pred, tooth, mask)

                    cc_target = torch.tensor(
                        [cusp_count_to_class(sample["cusp_count"])],
                        dtype=torch.long,
                    ).to(device, non_blocking=pin_memory)
                    loss_cc = F.cross_entropy(cc_logits, cc_target)

                    loss = (
                        loss_base
                        + loss_weights["cusp"] * loss_cusp
                        + loss_weights["surface"] * loss_surface
                        + loss_weights["cc"] * loss_cc
                    )

            loss_total = loss_total + loss
            update_metrics(metrics, pred, target, mask, cc_logits, cc_target)

        loss_total = loss_total / max(1, len(batch))

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

        metrics["loss_sum"] += loss_total.item()
        metrics["loss_count"] += 1

    return summarize_metrics(metrics)

def compute_arc_index_fallback(vertices):
    x = vertices[:, 0]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-8:
        return np.full((vertices.shape[0],), 0.5, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)

def load_mesh_any(mesh_or_path):
    mesh = mesh_or_path
    if isinstance(mesh_or_path, (str, Path)):
        mesh = trimesh.load_mesh(mesh_or_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = mesh.dump()
        if len(geoms) == 0:
            raise ValueError("No geometry found in mesh.")
        mesh = trimesh.util.concatenate(geoms)
    return mesh

def _load_tgn_vertices(mesh_path):
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        return np.asarray(mesh.vertices, dtype=np.float32)
    except Exception:
        mesh = load_mesh_any(mesh_path)
        mesh = mesh.copy()
        try:
            mesh.merge_vertices()
        except Exception:
            pass
        return np.asarray(mesh.vertices, dtype=np.float32)

def align_instances_to_mesh(mesh_path, instances, target_vertices, tgn_vertices=None):
    inst = np.asarray(instances, dtype=np.int64).reshape(-1)
    if tgn_vertices is None:
        inst_vertices = _load_tgn_vertices(mesh_path)
    else:
        inst_vertices = np.asarray(tgn_vertices, dtype=np.float32)
    if inst_vertices.shape[0] != inst.shape[0]:
        print("Warning: instance/vertex mismatch; unable to align instances.")
        return instances
    tree = cKDTree(inst_vertices)
    _, idx = tree.query(np.asarray(target_vertices, dtype=np.float32), k=1)
    return inst[idx]

def _normalize_mesh_path(p):
    try:
        return str(Path(p))
    except Exception:
        return str(p)

def _find_row_by_mesh_path(df, mesh_path):
    if df is None:
        return None
    mp = _normalize_mesh_path(mesh_path)
    obj_paths = df["obj_path"].apply(_normalize_mesh_path)
    match = df[obj_paths == mp]
    if len(match) > 0:
        return match.iloc[0]
    name = Path(mp).name
    match = df[obj_paths.apply(lambda p: Path(p).name) == name]
    if len(match) > 0:
        return match.iloc[0]
    return None

def build_vertex_adjacency(faces, n_verts):
    adj = [set() for _ in range(n_verts)]
    skipped = 0
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a < 0 or b < 0 or c < 0 or a >= n_verts or b >= n_verts or c >= n_verts:
            skipped += 1
            continue
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    if skipped > 0:
        print(f"Warning: skipped {skipped} faces with out-of-range indices.")
    return [list(s) for s in adj]

def _majority_label(vals, fallback=0):
    if len(vals) == 0:
        return fallback
    vals = np.asarray(vals, dtype=np.int64)
    if vals.size == 0:
        return fallback
    counts = np.bincount(vals)
    return int(np.argmax(counts))

def enforce_instance_majority(labels, instances, background=0):
    labels = np.asarray(labels, dtype=np.int64).copy()
    if instances is None:
        return labels
    inst = np.asarray(instances)
    if len(inst) != len(labels):
        return labels
    for inst_id in np.unique(inst):
        if inst_id == background:
            continue
        idx = np.where(inst == inst_id)[0]
        if idx.size == 0:
            continue
        vals = labels[idx]
        vals = vals[vals != background]
        if vals.size == 0:
            continue
        maj = _majority_label(vals, fallback=background)
        labels[idx] = maj
    return labels

def remove_small_components(labels, adj, min_size=200, background=0, valid_mask=None):
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = len(labels)
    visited = np.zeros(n, dtype=bool)
    for lbl in np.unique(labels):
        if lbl == background:
            continue
        idx = np.where(labels == lbl)[0]
        for v in idx:
            if visited[v]:
                continue
            if valid_mask is not None and not valid_mask[v]:
                visited[v] = True
                continue
            comp = []
            dq = [v]
            visited[v] = True
            while dq:
                u = dq.pop()
                comp.append(u)
                for w in adj[u]:
                    if visited[w]:
                        continue
                    if valid_mask is not None and not valid_mask[w]:
                        visited[w] = True
                        continue
                    if labels[w] == lbl:
                        visited[w] = True
                        dq.append(w)
            if len(comp) < min_size:
                labels[comp] = background
    return labels

def fill_holes(labels, adj, iters=2, background=0, valid_mask=None):
    labels = np.asarray(labels, dtype=np.int64).copy()
    for _ in range(iters):
        new = labels.copy()
        for i in range(len(labels)):
            if labels[i] != background:
                continue
            if valid_mask is not None and not valid_mask[i]:
                continue
            neigh = [labels[j] for j in adj[i] if labels[j] != background and (valid_mask is None or valid_mask[j])]
            if neigh:
                new[i] = _majority_label(neigh, fallback=background)
        labels = new
    return labels

def smooth_labels(labels, adj, iters=1, valid_mask=None, background=0, ignore_background=True):
    labels = np.asarray(labels, dtype=np.int64).copy()
    for _ in range(iters):
        new = labels.copy()
        for i in range(len(labels)):
            if valid_mask is not None and not valid_mask[i]:
                continue
            neigh = [labels[j] for j in adj[i] if valid_mask is None or valid_mask[j]]
            if ignore_background:
                neigh = [v for v in neigh if v != background]
            if neigh:
                new[i] = _majority_label(neigh, fallback=labels[i])
        labels = new
    return labels

def postprocess_instance_labels(
    instances,
    faces,
    min_component=200,
    fill_iters=2,
    smooth_iters=1,
    background=0,
):
    inst_in = np.asarray(instances, dtype=np.int64).copy()
    inst = inst_in.copy()
    faces = np.asarray(faces, dtype=np.int64)
    if inst.size == 0 or faces.size == 0:
        return inst
    adj = build_vertex_adjacency(faces, len(inst))
    if min_component > 0:
        inst = remove_small_components(inst, adj, min_size=min_component, background=background)
        if np.count_nonzero(inst) == 0:
            return inst_in
    inst = fill_holes(inst, adj, iters=fill_iters, background=background)
    inst = smooth_labels(inst, adj, iters=smooth_iters, background=background, ignore_background=True)
    if np.count_nonzero(inst) == 0:
        return inst_in
    return inst

def instances_from_semantic_labels(labels, faces, min_component=200, background=0):
    labels = np.asarray(labels, dtype=np.int64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    if labels.size == 0 or faces.size == 0:
        return labels
    adj = build_vertex_adjacency(faces, len(labels))
    inst = np.zeros_like(labels, dtype=np.int64)
    visited = np.zeros(len(labels), dtype=bool)
    next_id = 1
    for lbl in np.unique(labels):
        if lbl == background:
            continue
        idx = np.where(labels == lbl)[0]
        for v in idx:
            if visited[v]:
                continue
            comp = []
            stack = [v]
            visited[v] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adj[u]:
                    if visited[w]:
                        continue
                    if labels[w] == lbl:
                        visited[w] = True
                        stack.append(w)
            if min_component > 0 and len(comp) < min_component:
                continue
            inst[comp] = next_id
            next_id += 1
    return inst

def postprocess_fdi_labels(
    labels,
    faces,
    instances=None,
    min_component=200,
    fill_iters=2,
    smooth_iters=1,
    use_instance_majority=True,
    background=0,
    valid_mask=None,
    ignore_background=True,
):
    labels = np.asarray(labels, dtype=np.int64).copy()
    if use_instance_majority and instances is not None:
        labels = enforce_instance_majority(labels, instances, background=background)

    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return labels

    adj = build_vertex_adjacency(faces, len(labels))

    if min_component > 0:
        before = labels.copy()
        labels = remove_small_components(
            labels, adj, min_size=min_component, background=background, valid_mask=valid_mask
        )
        if np.count_nonzero(labels) == 0:
            labels = before

    labels = fill_holes(labels, adj, iters=fill_iters, background=background, valid_mask=valid_mask)

    before = labels.copy()
    labels = smooth_labels(
        labels,
        adj,
        iters=smooth_iters,
        valid_mask=valid_mask,
        background=background,
        ignore_background=ignore_background,
    )
    if np.count_nonzero(labels) == 0:
        labels = before

    return labels

def predict_fdi_for_mesh_full(mesh_or_path, jaw="lower", instances=None, n_points=4096, k=5, use_instances=True):
    mesh = load_mesh_any(mesh_or_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)

    N = vertices.shape[0]
    if N >= n_points:
        idxs = np.random.choice(N, n_points, replace=False)
    else:
        idxs = np.random.choice(N, n_points, replace=True)

    pts = vertices[idxs]
    if use_instances and instances is not None:
        if len(instances) != vertices.shape[0]:
            print("Warning: instances/vertices length mismatch; using fallback arc-index.")
            t_full = compute_arc_index_fallback(vertices)
        else:
            t_full = compute_arc_index_per_vertex(vertices, instances, jaw, axis=0)
    else:
        t_full = compute_arc_index_fallback(vertices)
    t = t_full[idxs][:, None]

    # DGCNN normalization (as in training) -- use full mesh stats
    center = vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(vertices - center, axis=1)) + 1e-8
    pts_norm = (pts - center) / scale
    pts4 = np.concatenate([pts_norm, t], axis=1)

    fdi_model, idx_map, _ = get_fdi_model()
    fdi_model.eval()
    with torch.no_grad():
        x = torch.from_numpy(pts4[None, ...]).float().to(device)
        logits = fdi_model(x)
        preds_idx = logits.argmax(dim=-1).cpu().numpy()[0]

    vec_map = np.vectorize(lambda i: idx_map[int(i)])
    preds_fdi_sampled = vec_map(preds_idx).astype(np.int64)

    tree = cKDTree(pts)
    _, nn_idx = tree.query(vertices, k=k)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    vote = preds_fdi_sampled[nn_idx]
    preds_fdi_full = np.apply_along_axis(lambda v: np.bincount(v).argmax(), 1, vote)

    return preds_fdi_full

def snap_points_to_surface(pred_pts, surface_pts):
    if surface_pts is None or len(surface_pts) == 0:
        return pred_pts
    tree = cKDTree(surface_pts)
    _, idx = tree.query(pred_pts, k=1)
    return surface_pts[idx]

def predict_landmarks_from_fdi(
    vertices_norm,
    preds_fdi_full,
    landmark_model,
    jaw_points=8192,
    tooth_points=2048,
    snap_to_surface=True,
    use_predicted_cusp_count=False,
):
    jaw_pts = _sample_points(vertices_norm, jaw_points)

    out_landmarks = []
    for fdi in np.unique(preds_fdi_full):
        if fdi <= 0:
            continue
        mask = preds_fdi_full == fdi
        if mask.sum() < 32:
            continue

        tooth_vertices = vertices_norm[mask]
        tooth_pts = _sample_points(tooth_vertices, tooth_points)

        center, scale = compute_tooth_norm_params(tooth_vertices)
        jaw_pts_local = apply_tooth_norm(jaw_pts, center, scale)
        tooth_pts_local = apply_tooth_norm(tooth_pts, center, scale)

        jaw_pts_t = torch.from_numpy(jaw_pts_local[None, ...]).float().to(device)
        tooth_pts_t = torch.from_numpy(tooth_pts_local[None, ...]).float().to(device)

        fdi_t = torch.tensor([float(fdi)], dtype=torch.float32).to(device)
        ttype_t = torch.tensor([float(tooth_type_from_fdi(fdi))], dtype=torch.float32).to(device)

        with torch.no_grad():
            pred, cc_logits = landmark_model(jaw_pts_t, tooth_pts_t, fdi_t, ttype_t)

        pred = pred[0].cpu().numpy()
        pred = invert_tooth_norm(pred, center, scale)
        if snap_to_surface:
            pred = snap_points_to_surface(pred, tooth_vertices)

        expected_cusps = expected_cusp_count(fdi)
        if use_predicted_cusp_count and expected_cusps > 0:
            cc = int(torch.argmax(cc_logits, dim=-1).cpu().item())
            cusp_count = min([0, 2, 4][cc], expected_cusps)
        else:
            cusp_count = expected_cusps

        for i, name in enumerate(BASE_ORDER):
            out_landmarks.append({
                "class": name,
                "coord": pred[i].tolist(),
                "fdi": int(fdi)
            })

        for i in range(cusp_count):
            out_landmarks.append({
                "class": f"Cusp{i+1}",
                "coord": pred[5 + i].tolist(),
                "fdi": int(fdi)
            })

    return out_landmarks

def predict_landmarks_from_mesh(
    mesh_or_path,
    jaw="lower",
    instances=None,
    landmark_model=None,
    use_instances=True,
    jaw_points=8192,
    tooth_points=2048,
    df_lookup=None,
    mask_gum=True,
    snap_to_surface=True,
    use_predicted_cusp_count=False,
    instances_are_semantic="auto",
    postprocess_fdi=True,
    min_fdi_component=50,
    fill_holes_iters=2,
    smooth_iters=1,
    use_instance_majority=True,
    postprocess_instances=True,
    instance_min_component=200,
    instance_fill_iters=2,
    instance_smooth_iters=1,
    jaw_ref_R=None,
    jaw_offset=None,
):
    if landmark_model is None:
        landmark_model = get_landmark_model()

    mesh = load_mesh_any(mesh_or_path)

    if instances is None and df_lookup is not None and isinstance(mesh_or_path, (str, Path)):
        row = _find_row_by_mesh_path(df_lookup, mesh_or_path)
        if row is not None:
            if instances is None and "instances" in row:
                instances = row["instances"]
            if "jaw" in row:
                jaw = row["jaw"]

    if instances is not None and len(instances) != len(mesh.vertices):
        instances = align_instances_to_mesh(mesh_or_path, instances, np.asarray(mesh.vertices, dtype=np.float32))
        print("Aligned instances to mesh vertices via nearest neighbor.")
    instances_for_arc = instances
    if instances_for_arc is not None:
        if instances_are_semantic == "auto":
            is_sem = len(np.unique(instances_for_arc)) <= 20
        else:
            is_sem = bool(instances_are_semantic)
        if is_sem:
            instances_for_arc = instances_from_semantic_labels(
                instances_for_arc,
                np.asarray(mesh.faces, dtype=np.int64),
                min_component=instance_min_component,
                background=0,
            )
    if instances_for_arc is not None and postprocess_instances:
        instances_for_arc = postprocess_instance_labels(
            instances_for_arc,
            np.asarray(mesh.faces, dtype=np.int64),
            min_component=instance_min_component,
            fill_iters=instance_fill_iters,
            smooth_iters=instance_smooth_iters,
            background=0,
        )
    if instances_for_arc is not None:
        inst_arr = np.asarray(instances_for_arc)
        if inst_arr.size == 0 or np.count_nonzero(inst_arr) == 0 or np.unique(inst_arr).size <= 1:
            instances_for_arc = None

    use_instances = use_instances and instances_for_arc is not None

    preds_fdi_full = predict_fdi_for_mesh_full(
        mesh,
        jaw=jaw,
        instances=instances_for_arc,
        n_points=4096,
        k=5,
        use_instances=use_instances,
    )

    valid_mask = None
    if mask_gum and instances_for_arc is not None and len(instances_for_arc) == len(preds_fdi_full):
        preds_fdi_full = preds_fdi_full.copy()
        preds_fdi_full[instances_for_arc == 0] = 0
        valid_mask = np.asarray(instances_for_arc) > 0

    if postprocess_fdi:
        preds_fdi_full = postprocess_fdi_labels(
            preds_fdi_full,
            np.asarray(mesh.faces, dtype=np.int64),
            instances=instances_for_arc,
            min_component=min_fdi_component,
            fill_iters=fill_holes_iters,
            smooth_iters=smooth_iters,
            use_instance_majority=use_instance_majority,
            background=0,
            valid_mask=valid_mask,
        )

    vertices_raw = np.asarray(mesh.vertices, dtype=np.float64)
    vertices_norm, _, T_jaw = apply_jaw_normalization(vertices_raw, ldmrks=[], ref_R=jaw_ref_R, offset=jaw_offset)

    mesh_norm = mesh.copy()
    mesh_norm.apply_transform(T_jaw)

    pred_landmarks = predict_landmarks_from_fdi(
        vertices_norm,
        preds_fdi_full,
        landmark_model,
        jaw_points=jaw_points,
        tooth_points=tooth_points,
        snap_to_surface=snap_to_surface,
        use_predicted_cusp_count=use_predicted_cusp_count,
    )

    return preds_fdi_full, pred_landmarks, mesh_norm

def load_toothgroupnet_predictions(mesh_path, model_name="tgnet", out_path=None):
    if out_path is None:
        npz_path = Path(mesh_path).with_name(f"{Path(mesh_path).stem}_toothgroupnet_{model_name}.npz")
    else:
        npz_path = Path(out_path)

    if npz_path.exists():
        data = np.load(npz_path)
        labels = data["labels"].astype(np.int64)
        instances = data["instances"].astype(np.int64)
        tgn_vertices = data["tgn_vertices"].astype(np.float32) if "tgn_vertices" in data else None
        return labels, instances, npz_path, tgn_vertices
    return None, None, npz_path, None

def run_toothgroupnet_inference(
    mesh_path,
    model_name="tgnet",
    checkpoint_dir=Path("checkpoints") / "ckpts(new)",
    script_path=Path("toothgroupnet_single_infer.py"),
    out_path=None,
    rebase=False,
    rebase_out=None,
    flip_x=False,
    flip_y=False,
    flip_z=False,
):
    import subprocess
    import sys

    mesh_path = Path(mesh_path)
    if out_path is None:
        out_path = mesh_path.with_name(f"{mesh_path.stem}_toothgroupnet_{model_name}.npz")

    cmd = [
        sys.executable,
        str(script_path),
        "--mesh",
        str(mesh_path),
        "--model",
        str(model_name),
        "--ckpt-dir",
        str(checkpoint_dir),
        "--out",
        str(out_path),
    ]
    if rebase:
        cmd.append("--rebase")
        if rebase_out is not None:
            cmd.extend(["--rebase-out", str(rebase_out)])
        if flip_x:
            cmd.append("--flip-x")
        if flip_y:
            cmd.append("--flip-y")
        if flip_z:
            cmd.append("--flip-z")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    data = np.load(out_path)
    labels = data["labels"].astype(np.int64)
    instances = data["instances"].astype(np.int64)
    tgn_vertices = data["tgn_vertices"].astype(np.float32) if "tgn_vertices" in data else None
    return labels, instances, out_path, tgn_vertices


def make_color_map(values, background=(200, 200, 200, 255)):
    """Generate a repeatable color map for integer labels."""
    uniq = [int(v) for v in sorted(set(np.asarray(values).flatten()))]
    color_map = {}
    for i, val in enumerate(uniq):
        if val <= 0:
            color_map[val] = background
            continue
        hue = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.95)
        color_map[val] = (int(r * 255), int(g * 255), int(b * 255), 255)
    return color_map


def color_mesh_with_map(mesh, values, color_map):
    """Return a copy of mesh with vertex colors according to color_map."""
    colored = mesh.copy()
    values_arr = np.asarray(values, dtype=np.int64)
    if len(values_arr) != len(colored.vertices):
        raise ValueError("values length must match number of vertices")
    vertex_colors = np.zeros((len(values_arr), 4), dtype=np.uint8)
    for i, v in enumerate(values_arr):
        vertex_colors[i] = color_map.get(int(v), (180, 180, 180, 255))
    colored.visual.vertex_colors = vertex_colors
    return colored


LANDMARK_COLORS = {
    "Mesial": (255, 0, 0, 255),
    "Distal": (0, 120, 255, 255),
    "InnerPoint": (0, 200, 160, 255),
    "OuterPoint": (200, 0, 200, 255),
    "FacialPoint": (255, 165, 0, 255),
    "Cusp1": (0, 255, 0, 255),
    "Cusp2": (0, 180, 0, 255),
    "Cusp3": (0, 140, 0, 255),
    "Cusp4": (0, 100, 0, 255),
}


def landmark_spheres(landmarks, *, radius=0.3, default_color=(0, 0, 0, 255)):
    """Create trimesh spheres for each landmark with class-based colors."""
    spheres = []
    for lm in landmarks:
        p = np.array(lm["coord"], dtype=np.float32)
        color = LANDMARK_COLORS.get(lm.get("class"), default_color)
        sph = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sph.apply_translation(p)
        sph.visual.vertex_colors = color
        spheres.append(sph)
    return spheres


def run_full_pipeline(
    mesh_path,
    jaw="lower",
    *,
    tgn_model_name="tgnet",
    tgn_ckpt_dir=Path("checkpoints") / "ckpts(new)",
    use_existing_npz=True,
    run_toothgroupnet=True,
    tgn_flip_y=False,
    tgn_flip_z=False,
    mask_gum=True,
    jaw_offset=None,
    num_path_points=300,
    path_smoothing=0.0,
    instance_min_component=None,
    instance_fill_iters=None,
    instance_smooth_iters=None,
    z_rot_step_deg=10.0,
    z_lift_step=0.5,
    max_z_lifts=40,
    ref_mesh_path=None,
    jaw_ref_R=None,
    df_lookup=None,
    stl_decimate_target_faces=None,
    stl_decimate_out=None,
    force_stl_box=False,
):
    """
    End-to-end wrapper for segmentation, labeling, landmarking, and trajectory generation.

    Returns a dict with meshes, labels, landmarks, paths, and scanner poses.
    """
    mesh_path = Path(mesh_path)
    is_stl_input = mesh_path.suffix.lower() == ".stl"

    # STL-specific decimation (optional)
    if is_stl_input and stl_decimate_target_faces:
        try:
            mesh_dec = load_mesh_any(mesh_path)
            mesh_dec = mesh_dec.simplify_quadratic_decimation(int(stl_decimate_target_faces))
            if stl_decimate_out is None:
                stl_decimate_out = mesh_path.with_name(f"{mesh_path.stem}_decimated.obj")
            mesh_dec.export(stl_decimate_out)
            mesh_path = Path(stl_decimate_out)
        except Exception as exc:
            print(f"Warning: STL decimation failed: {exc}")

    # Auto-rebase STL, keep OBJ
    tgn_rebase = is_stl_input

    npz_out = mesh_path.with_name(f"{mesh_path.stem}_toothgroupnet_{tgn_model_name}.npz")
    labels = instances = tgn_vertices = None

    if use_existing_npz:
        labels, instances, npz_path, tgn_vertices = load_toothgroupnet_predictions(
            mesh_path,
            model_name=tgn_model_name,
            out_path=npz_out,
        )
    else:
        npz_path = npz_out

    rebase_out = mesh_path.with_name(f"{mesh_path.stem}_rebased.obj") if tgn_rebase else None
    if instances is None and run_toothgroupnet:
        labels, instances, npz_path, tgn_vertices = run_toothgroupnet_inference(
            mesh_path,
            model_name=tgn_model_name,
            checkpoint_dir=tgn_ckpt_dir,
            out_path=npz_path,
            rebase=tgn_rebase,
            rebase_out=rebase_out,
            flip_y=tgn_flip_y,
            flip_z=tgn_flip_z,
        )

    mesh_path_for_landmarks = mesh_path
    if tgn_rebase and rebase_out is not None and rebase_out.exists():
        mesh_path_for_landmarks = rebase_out
    elif tgn_rebase and rebase_out is not None and not rebase_out.exists():
        print(f"Warning: expected rebased mesh at {rebase_out} but it was not found; using original mesh.")

    mesh_in = load_mesh_any(mesh_path_for_landmarks)

    jaw_ref_R_use = jaw_ref_R
    if jaw_ref_R_use is None and ref_mesh_path is not None:
        jaw_ref_R_use = load_reference_rotation(ref_mesh_path, df_lookup=df_lookup)

    # Set defaults that differ between OBJ and STL pipelines
    if jaw_offset is None and is_stl_input:
        jaw_offset = np.array([0.0, 0.0, 5.0], dtype=np.float64)

    imc = instance_min_component
    fill_it = instance_fill_iters
    smooth_it = instance_smooth_iters
    if imc is None:
        imc = 0 if is_stl_input else 5
    if fill_it is None:
        fill_it = 2
    if smooth_it is None:
        smooth_it = 3 if is_stl_input else 1

    instances_clean = None
    if instances is not None:
        if len(instances) != len(mesh_in.vertices):
            instances = align_instances_to_mesh(
                mesh_path_for_landmarks,
                instances,
                np.asarray(mesh_in.vertices, dtype=np.float32),
                tgn_vertices=tgn_vertices,
            )
        inst_for_arc = instances
        faces_arr = np.asarray(mesh_in.faces, dtype=np.int64)
        if len(np.unique(inst_for_arc)) <= 20:
            inst_for_arc = instances_from_semantic_labels(
                inst_for_arc,
                faces_arr,
                min_component=imc,
                background=0,
            )
        instances_clean = postprocess_instance_labels(
            inst_for_arc,
            faces_arr,
            min_component=imc,
            fill_iters=fill_it,
            smooth_iters=smooth_it,
            background=0,
        )
        if np.count_nonzero(instances_clean) == 0:
            instances_clean = inst_for_arc
        inst_arr = np.asarray(instances_clean)
        if inst_arr.size == 0 or np.count_nonzero(inst_arr) == 0 or np.unique(inst_arr).size <= 1:
            instances_clean = None

    landmark_model = get_landmark_model()
    preds_fdi_full, pred_landmarks, mesh_norm = predict_landmarks_from_mesh(
        mesh_path_for_landmarks,
        jaw=jaw,
        instances=instances_clean,
        landmark_model=landmark_model,
        use_instances=True,
        df_lookup=df_lookup,
        mask_gum=mask_gum,
        instances_are_semantic=False,
        postprocess_instances=False,
        jaw_offset=jaw_offset,
        jaw_ref_R=jaw_ref_R_use,
    )

    verts_norm = np.asarray(mesh_norm.vertices, dtype=np.float64)
    upper = str(jaw).lower() == "upper"
    keypoints, projections, normals = compute_keypoints_ordered(
        pred_landmarks, mesh_norm, verts_norm, preds_fdi_full, upper=upper
    )
    waypoints = [kp for kp in keypoints if kp is not None]
    if waypoints:
        path_pts = generate_smooth_path(waypoints, num_points=num_path_points, smoothing=path_smoothing)
    else:
        path_pts = np.empty((0, 3), dtype=float)
    path_pts_original = np.asarray(path_pts)

    normals_valid = [n for n in normals if n is not None]
    if normals_valid:
        rot_mats = [normal_to_rotation_matrix(n) for n in normals_valid]
    else:
        rot_mats = [np.eye(3)]

    base_quats = np.array([R.from_matrix(r).as_quat() for r in rot_mats])
    if len(path_pts) == 0:
        interp_rots = []
    elif len(base_quats) == 1:
        interp_rots = [R.from_quat(base_quats[0]) for _ in range(len(path_pts))]
    else:
        interp_func = interp1d(np.linspace(0, 1, len(base_quats)), base_quats.T, kind="linear", fill_value="extrapolate")
        interp_quats = interp_func(np.linspace(0, 1, len(path_pts))).T
        interp_rots = [R.from_quat(q) for q in interp_quats]

    bmin, bmax = mesh_norm.bounds
    size_x = float(bmax[0] - bmin[0])
    size_y = float(bmax[1] - bmin[1])
    size_z = float(bmax[2] - bmin[2])

    # Mouth box sizing differs by format (mirrors notebook branches)
    use_stl_box = force_stl_box or is_stl_input
    if use_stl_box:
        mouth_box = create_open_mouth_box(size=(size_x + 10, size_y + 5, 3 * size_z), thickness=2.0, opening_axis="+Y")
        jaw_center_xy = np.array([(bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5], dtype=np.float64)
        box_center = np.array([jaw_center_xy[0], jaw_center_xy[1] - 10, (3.5 * size_z) * 0.5], dtype=np.float64)
    else:
        # OBJ branch from landmarks_prep copy 5
        mouth_box = create_open_mouth_box(size=(size_x, size_y, 1.5 * size_z), thickness=2.0, opening_axis="+Y")
        jaw_center_xy = np.array([(bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5], dtype=np.float64)
        box_center = np.array([jaw_center_xy[0], jaw_center_xy[1] - 10, (2.0 * size_z) * 0.5], dtype=np.float64)
    mouth_box.apply_translation(box_center)

    jaw_in_box = trimesh.util.concatenate([mesh_norm, mouth_box])
    jaw_in_box.fix_normals()

    scanner_mesh = create_scanner_mesh(SCAN_FIELD_DIM, OVERALL_FIELD_DIM)
    if len(path_pts) == 0 or len(interp_rots) == 0:
        scanner_points = np.empty((0, 3), dtype=float)
        scanner_rots = []
    else:
        scanner_points, scanner_rots, kept_idx = place_scanner_along_path_interp_rots(
            path_pts,
            interp_rots,
            scanner_mesh,
            obstacles=[jaw_in_box],
            z_rot_step_deg=z_rot_step_deg,
            z_lift_step=z_lift_step,
            max_z_lifts=max_z_lifts,
            scan_dim=SCAN_FIELD_DIM,
            debug=False,
        )
        # Drop any original path points that failed to get a scanner pose
        if len(scanner_points) != len(path_pts):
            kept_idx = np.asarray(kept_idx, dtype=int)
            scanner_points = np.asarray(scanner_points)
            scanner_rots = list(scanner_rots)
            path_pts = np.asarray(path_pts)[kept_idx]
            interp_rots = [interp_rots[k] for k in kept_idx]
        # Use placed scanner points/rots as the new path to avoid gaps
        path_pts = np.asarray(scanner_points)
        interp_rots = list(scanner_rots)

    return {
        "mesh_norm": mesh_norm,
        "instances": instances_clean,
        "preds_fdi_full": preds_fdi_full,
        "pred_landmarks": pred_landmarks,
        "keypoints": keypoints,
        "projections": projections,
        "normals": normals,
        "path_pts": path_pts,
        "path_pts_original": path_pts_original,
        "interp_rots": interp_rots,
        "mouth_box": mouth_box,
        "jaw_in_box": jaw_in_box,
        "scanner_mesh": scanner_mesh,
        "scanner_points": scanner_points,
        "scanner_rotations": scanner_rots,
        "labels": labels,
        "tgn_vertices": tgn_vertices,
        "jaw_ref_R": jaw_ref_R_use,
        "mesh_path_for_landmarks": mesh_path_for_landmarks,
        "rebased_mesh_path": rebase_out if tgn_rebase else None,
        "is_stl_input": is_stl_input,
        "tgn_rebase_used": bool(tgn_rebase),
    }
