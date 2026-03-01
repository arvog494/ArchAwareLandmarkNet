"""
setup_data.py — Rebuild __data__/3DTeethLand_combined/ and __data__/landmarks/
from the 3DTeethLand OSF download.

The OSF dataset is published as two separate downloads (meshes + landmarks),
each split into train and test archives. Once extracted, run this script to
merge them into the flat structure expected by the pipeline.

Usage
-----
    python setup_data.py \
        --meshes-train  path/to/3DTeethSeg22_challenge_train \
        --meshes-test   path/to/3DTeethSeg22_challenge_test \
        --landmarks-train path/to/3DTeethLand_landmarks_train \
        --landmarks-test  path/to/3DTeethLand_landmarks_test

Expected input structure (from OSF)
------------------------------------
    3DTeethSeg22_challenge_train/      3DTeethLand_landmarks_train/
    ├── lower/                         ├── lower/
    │   ├── {ID}/                      │   ├── {ID}/
    │   │   ├── {ID}_lower.obj         │   │   └── {ID}_lower__kpt.json
    │   │   └── {ID}_lower.json        │   └── ...
    │   └── ...                        └── upper/
    └── upper/                             └── ...
        └── ...

Output (created inside __data__/)
---------------------------------
    __data__/3DTeethLand_combined/      __data__/landmarks/
    ├── lower/                          ├── lower/
    │   ├── {ID}/                       │   ├── {ID}/
    │   │   ├── {ID}_lower.obj          │   │   └── {ID}_lower__kpt.json
    │   │   └── {ID}_lower.json         │   └── ...
    │   └── ...                         └── upper/
    └── upper/                              └── ...
        └── ...
"""

import argparse
import shutil
import sys
from pathlib import Path


DATA_ROOT = Path(__file__).parent / "__data__"


def merge_patient_folders(src: Path, dst: Path, label: str) -> int:
    """Copy all patient sub-folders from *src* into *dst*, preserving structure.

    Returns the number of patient folders copied.
    """
    if not src.is_dir():
        print(f"  [SKIP] {src} does not exist")
        return 0

    count = 0
    for patient_dir in sorted(src.iterdir()):
        if not patient_dir.is_dir():
            continue
        target = dst / patient_dir.name
        if target.exists():
            # Merge files into existing folder (don't overwrite)
            for f in patient_dir.iterdir():
                dest_file = target / f.name
                if not dest_file.exists():
                    shutil.copy2(f, dest_file)
        else:
            shutil.copytree(patient_dir, target)
        count += 1

    print(f"  {label}: {count} patient folders -> {dst.relative_to(DATA_ROOT.parent)}")
    return count


def setup_meshes(train_path: Path, test_path: Path) -> None:
    """Merge train + test mesh folders into __data__/3DTeethLand_combined/."""
    print("\n=== Meshes ===")
    combined = DATA_ROOT / "3DTeethLand_combined"
    total = 0

    for jaw in ("lower", "upper"):
        out = combined / jaw
        out.mkdir(parents=True, exist_ok=True)
        total += merge_patient_folders(train_path / jaw, out, f"train/{jaw}")
        total += merge_patient_folders(test_path / jaw, out, f"test/{jaw}")

    print(f"  Total: {total} patient folders in 3DTeethLand_combined/")


def setup_landmarks(train_path: Path, test_path: Path) -> None:
    """Merge train + test landmark folders into __data__/landmarks/."""
    print("\n=== Landmarks ===")
    landmarks = DATA_ROOT / "landmarks"
    total = 0

    for jaw in ("lower", "upper"):
        out = landmarks / jaw
        out.mkdir(parents=True, exist_ok=True)
        total += merge_patient_folders(train_path / jaw, out, f"train/{jaw}")
        total += merge_patient_folders(test_path / jaw, out, f"test/{jaw}")

    print(f"  Total: {total} patient folders in landmarks/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild __data__/ from OSF downloads (3DTeethLand challenge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--meshes-train",
        type=Path,
        required=True,
        help="Path to the extracted mesh training folder (e.g. 3DTeethSeg22_challenge_train/)",
    )
    parser.add_argument(
        "--meshes-test",
        type=Path,
        required=True,
        help="Path to the extracted mesh testing folder (e.g. 3DTeethSeg22_challenge_test/)",
    )
    parser.add_argument(
        "--landmarks-train",
        type=Path,
        required=True,
        help="Path to the extracted landmarks training folder (e.g. 3DTeethLand_landmarks_train/)",
    )
    parser.add_argument(
        "--landmarks-test",
        type=Path,
        required=True,
        help="Path to the extracted landmarks testing folder (e.g. 3DTeethLand_landmarks_test/)",
    )
    args = parser.parse_args()

    # Validate inputs
    ok = True
    for name, path in [
        ("meshes-train", args.meshes_train),
        ("meshes-test", args.meshes_test),
        ("landmarks-train", args.landmarks_train),
        ("landmarks-test", args.landmarks_test),
    ]:
        if not path.is_dir():
            print(f"ERROR: --{name} path does not exist: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)

    print(f"Data root: {DATA_ROOT.resolve()}")

    setup_meshes(args.meshes_train, args.meshes_test)
    setup_landmarks(args.landmarks_train, args.landmarks_test)

    print("\nDone! You can now run the notebooks.")
    print("  - final_pipeline.ipynb       (full demo)")
    print("  - fdi_model_training.ipynb   (train FDI models)")
    print("  - landmarks_preparation.ipynb (landmark datasets)")


if __name__ == "__main__":
    main()
