from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


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

    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a, axis))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    k = v / s
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3) + K * s + (K @ K) * (1.0 - c)
    return R


def _compute_jaw_normalization_T(vertices, flip_x=False, flip_y=False, flip_z=False):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("vertices must have shape (N,3)")

    origin = vertices.mean(axis=0)
    x = vertices - origin
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    normal_axis = vh[-1]
    R_level = _rotation_matrix_from_vectors(normal_axis, np.array([0.0, 0.0, 1.0]))

    v1 = (R_level @ (vertices - origin).T).T

    xy = v1[:, :2] - np.mean(v1[:, :2], axis=0)
    if xy.shape[0] >= 3:
        _, _, vh2 = np.linalg.svd(xy, full_matrices=False)
        x2d = vh2[0]
    else:
        x2d = np.array([1.0, 0.0])

    x_axis = _normalize(np.array([x2d[0], x2d[1], 0.0]))
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))

    if flip_x:
        x_axis = -x_axis
    if flip_y:
        y_axis = -y_axis
    if flip_z:
        z_axis = -z_axis

    B = np.stack([x_axis, y_axis, z_axis], axis=1)
    R_inplane = B.T

    R = R_inplane @ R_level
    t0 = -R @ origin

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


def _load_mesh_any(path: Path):
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = mesh.dump()
        if len(geoms) == 0:
            raise ValueError("No geometry found in mesh.")
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def _load_tgn_vertices(path: Path) -> np.ndarray:
    try:
        import open3d as o3d

        o3d_mesh = o3d.io.read_triangle_mesh(str(path))
        if o3d_mesh.is_empty():
            raise ValueError("Open3D returned an empty mesh.")
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()
        return np.asarray(o3d_mesh.vertices)
    except Exception:
        mesh = _load_mesh_any(path)
        mesh.merge_vertices()
        return np.asarray(mesh.vertices)


def _rebase_mesh(path: Path, out_path: Path | None, flip_x=False, flip_y=False, flip_z=False) -> Path:
    mesh = _load_mesh_any(path)
    T = _compute_jaw_normalization_T(np.asarray(mesh.vertices), flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
    mesh.apply_transform(T)

    if out_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".obj", prefix=f"{path.stem}_rebased_")
        Path(tmp_path).unlink(missing_ok=True)
        out_path = Path(tmp_path)

    # Deduplicate vertices so output length matches TGNet inference mesh.
    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(str(out_path), o3d_mesh)
    except Exception:
        mesh.merge_vertices()
        mesh.export(out_path)
    return out_path


def _ensure_tgn_on_path(tgn_root: Path) -> None:
    tgn_root = tgn_root.resolve()
    if not tgn_root.exists():
        raise FileNotFoundError(f"ToothGroupNetwork not found: {tgn_root}")
    if str(tgn_root) not in sys.path:
        sys.path.insert(0, str(tgn_root))


def _set_eval(pipeline) -> None:
    if hasattr(pipeline, "model"):
        pipeline.model.eval()
    if hasattr(pipeline, "first_module"):
        pipeline.first_module.eval()
    if hasattr(pipeline, "bdl_module"):
        pipeline.bdl_module.eval()


def _build_pipeline(model_name: str, ckpt_dir: Path):
    _ensure_tgn_on_path(Path("ToothGroupNetwork"))
    from inference_pipelines.inference_pipeline_maker import make_inference_pipeline

    ckpt_dir = ckpt_dir.resolve()
    if model_name == "tgnet":
        ckpt_paths = [ckpt_dir / "tgnet_fps.h5", ckpt_dir / "tgnet_bdl.h5"]
    else:
        ckpt_paths = [ckpt_dir / f"{model_name}.h5"]

    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing checkpoint(s): {missing_str}")

    pipeline = make_inference_pipeline(model_name, [str(p) for p in ckpt_paths])
    _set_eval(pipeline)
    return pipeline


def run_inference(
    mesh_path: Path,
    model_name: str,
    ckpt_dir: Path,
    out_path: Path | None,
    rebase: bool = False,
    rebase_out: Path | None = None,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> Path:
    mesh_path = mesh_path.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ToothGroupNetwork inference.")

    temp_path = None
    infer_path = mesh_path
    if rebase:
        temp_path = _rebase_mesh(mesh_path, rebase_out, flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
        infer_path = temp_path

    pipeline = _build_pipeline(model_name, ckpt_dir)
    pred = pipeline(str(infer_path))
    labels = pred["sem"].astype(np.int64)
    instances = pred["ins"].astype(np.int64)
    tgn_vertices = _load_tgn_vertices(infer_path)
    if tgn_vertices.dtype != np.float32:
        tgn_vertices = tgn_vertices.astype(np.float32)

    if out_path is None:
        out_path = mesh_path.with_name(f"{mesh_path.stem}_toothgroupnet_{model_name}.npz")
    else:
        out_path = out_path.resolve()

    np.savez(out_path, labels=labels, instances=instances, tgn_vertices=tgn_vertices)
    if rebase and temp_path is not None and rebase_out is None:
        try:
            temp_path.unlink()
        except Exception:
            pass
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ToothGroupNetwork on a single mesh. "
            "No axis reorientation is applied; your mesh must already be aligned."
        )
    )
    parser.add_argument("--mesh", required=True, help="Path to a single .obj mesh")
    parser.add_argument(
        "--model",
        default="tgnet",
        help="Model name: tgnet | tsegnet | pointnet | pointnetpp | dgcnn | pointtransformer",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=str(Path("checkpoints") / "ckpts(new)"),
        help="Checkpoint directory containing .h5 files",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output .npz path (default: next to mesh, *_toothgroupnet_<model>.npz)",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="Rebase mesh axes before inference (uses notebook-style normalization).",
    )
    parser.add_argument(
        "--rebase-out",
        default="",
        help="Optional output path for rebased mesh (default: temp .obj).",
    )
    parser.add_argument("--flip-x", action="store_true", help="Flip X axis during rebase.")
    parser.add_argument("--flip-y", action="store_true", help="Flip Y axis during rebase.")
    parser.add_argument("--flip-z", action="store_true", help="Flip Z axis during rebase.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out) if args.out else None
    rebase_out = Path(args.rebase_out) if args.rebase_out else None

    if args.rebase:
        print("Rebasing mesh to jaw orientation before inference.")
    else:
        print("Note: input mesh must already be axis-aligned for ToothGroupNetwork.")
    print("Running ToothGroupNetwork inference...")
    out_path = run_inference(
        mesh_path,
        args.model,
        ckpt_dir,
        out_path,
        rebase=args.rebase,
        rebase_out=rebase_out,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
