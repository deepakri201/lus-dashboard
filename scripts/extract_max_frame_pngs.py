#!/usr/bin/env python3
"""
Extract max-annotation frames from LUS DICOM clips as PNGs.

Mirrors completed-assignments-all layout under pngs/:
  pngs/phase-N/annotator-folder/case-id/{file_hash}_{metric}.png  (only if annotated)
  Metrics: _pleura, _discrete, _confluent — e.g. af4d429e56_3d785a24_pleura.png,
  af4d429e56_3d785a24_discrete.png, af4d429e56_3d785a24_confluent.png
  pngs/phase-N/annotator-folder/case-id/grid_{metric}.png

Metrics: pleura, discrete, confluent (max frame per metric from frame_annotations).
Clips with no frame_annotations are skipped entirely. Grid cells without a
metric annotation are left blank.

Usage:
  pip install -r requirements.txt
  python scripts/extract_max_frame_pngs.py --test          # default phase-3 test clip
  python scripts/extract_max_frame_pngs.py --test --test-dcm path/to/clip.dcm
  python scripts/extract_max_frame_pngs.py                 # full run
  python scripts/extract_max_frame_pngs.py --phase phase-5 --jobs 4 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import pydicom
    from pydicom.dataset import Dataset
except ImportError:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise

# Enable JPEG/JPEG-2000 compressed DICOM (common for ultrasound cine clips).
try:
    import pylibjpeg  # noqa: F401
except ImportError:
    pass

LUNG_ZONE_ORDER = [
    "right anterior upper (1)",
    "right anterior lower (2)",
    "right lateral upper (3)",
    "right lateral lower (4)",
    "left anterior upper (5)",
    "left anterior lower (6)",
    "left lateral upper (7)",
    "left lateral lower (8)",
]

METRICS = ("pleura", "discrete", "confluent")
DEFAULT_CELL_SIZE = (320, 240)
GRID_BG = (15, 18, 24)

# Default clip for --test (phase-3 andrew, right lateral lower, frames 57 & 77)
DEFAULT_TEST_DCM_REL = (
    "phase-3/andrew-r21-lahey-10-b-line-quant-assignment-full-scale-phase-3/"
    "1a7bed8054c0/b5ed3d4afc_0f49356d.dcm"
)


def pick_best(
    best: Tuple[float, Optional[str]], value: float, frame_num: str
) -> Tuple[float, Optional[str]]:
    if value < 0:
        return best
    if best[1] is None or value > best[0]:
        return (value, str(frame_num))
    return best


def max_frames_from_json(data: dict) -> Optional[Dict[str, Optional[str]]]:
    """Return max frame number per metric, or None if clip has no annotations."""
    frames = data.get("frame_annotations") or {}
    if not isinstance(frames, dict) or not frames:
        return None

    pleura_best = (-1.0, None)
    discrete_best = (-1.0, None)
    confluent_best = (-1.0, None)
    saw_frame = False

    for frame_num, frame in frames.items():
        if not frame or frame.get("type") not in (None, "manual_lus_score"):
            continue
        saw_frame = True

        b_lines = frame.get("b_line_count") or {}
        if frame.get("pleura_percentage") is not None:
            pleura_best = pick_best(pleura_best, float(frame["pleura_percentage"]), frame_num)
        if b_lines.get("discrete") is not None:
            discrete_best = pick_best(discrete_best, float(b_lines["discrete"]), frame_num)
        if b_lines.get("confluent") is not None:
            confluent_best = pick_best(confluent_best, float(b_lines["confluent"]), frame_num)

    if not saw_frame:
        return None

    result = {
        "pleura": pleura_best[1],
        "discrete": discrete_best[1],
        "confluent": confluent_best[1],
    }
    if not any(result.values()):
        return None
    return result


def normalize_to_uint8(pixels: np.ndarray) -> np.ndarray:
    arr = pixels.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_dicom(dcm_path: Path) -> Tuple[Optional[Dataset], Optional[np.ndarray], Optional[str]]:
    try:
        import pylibjpeg  # noqa: F401 — registers JPEG handlers for pydicom
    except ImportError:
        pass
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array
        if arr is None or (hasattr(arr, "size") and arr.size == 0):
            return None, None, f"{dcm_path.name}: empty pixel array"
        return ds, arr, None
    except Exception as exc:
        hint = ""
        if "jpeg" in str(exc).lower() or "decoder" in str(exc).lower():
            hint = " (try: pip install pylibjpeg pylibjpeg-libjpeg)"
        return None, None, f"{dcm_path.name}: {exc}{hint}"


def resolve_frame_index(frame_num: str, n_frames: int) -> int:
    """Map annotation frame label to 0-based DICOM index (prefer 0-based, fall back to 1-based)."""
    idx = int(frame_num)
    if n_frames <= 1:
        return 0
    if 0 <= idx < n_frames:
        return idx
    if 1 <= idx <= n_frames:
        return idx - 1
    return min(max(idx, 0), n_frames - 1)


def grayscale_frame_2d(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        if frame.shape[-1] == 1:
            return frame[..., 0]
        if frame.shape[-1] >= 3:
            return np.mean(frame[..., :3], axis=-1)
    return None


def dicom_rows_cols(
    ds: Dataset, rows_hint: Optional[int] = None, cols_hint: Optional[int] = None
) -> Tuple[int, int]:
    rows = int(rows_hint or getattr(ds, "Rows", 0) or 0)
    cols = int(cols_hint or getattr(ds, "Columns", 0) or 0)
    return rows, cols


def count_frames(ds: Dataset, arr: np.ndarray, rows: int, cols: int) -> int:
    tagged = int(getattr(ds, "NumberOfFrames", 0) or 0)
    if tagged > 1:
        return tagged
    if arr.ndim == 2:
        return 1
    if arr.ndim == 4:
        return arr.shape[0]
    if arr.ndim == 3:
        if rows and cols:
            if arr.shape[0] == rows and arr.shape[1] == cols:
                return arr.shape[2]
            if arr.shape[1] == rows and arr.shape[2] == cols:
                return arr.shape[0]
        # Heuristic: shortest axis is often frame count for cine loops
        if min(arr.shape) < max(arr.shape) / 8:
            return min(arr.shape)
        return arr.shape[0]
    return 1


def extract_via_pydicom_frame_api(ds: Dataset, frame_num: str, n_frames: int) -> Optional[np.ndarray]:
    """Decode one frame directly from DICOM (best for JPEG multiframe)."""
    candidates = []
    for idx in (
        resolve_frame_index(frame_num, n_frames),
        int(frame_num),
        int(frame_num) - 1,
    ):
        if idx not in candidates and idx >= 0:
            candidates.append(idx)

    # pydicom 3.x: pixel_array(ds, index=i)
    try:
        from pydicom.pixels import pixel_array as pydicom_pixel_array

        for idx in candidates:
            try:
                frame = pydicom_pixel_array(ds, index=idx)
                gray = grayscale_frame_2d(np.asarray(frame))
                if gray is not None:
                    return gray
            except TypeError:
                break
            except Exception:
                continue
    except ImportError:
        pass

    # pydicom 3.x: get_frame(ds, index)
    try:
        from pydicom.pixels import get_frame

        for idx in candidates:
            try:
                frame = get_frame(ds, idx)
                gray = grayscale_frame_2d(np.asarray(frame))
                if gray is not None:
                    return gray
            except Exception:
                continue
    except ImportError:
        pass

    return None


def extract_frame_pixels(
    ds: Dataset,
    arr: np.ndarray,
    frame_num: str,
    rows_hint: Optional[int] = None,
    cols_hint: Optional[int] = None,
) -> Optional[np.ndarray]:
    rows, cols = dicom_rows_cols(ds, rows_hint, cols_hint)
    n_frames = count_frames(ds, arr, rows, cols)

    frame = extract_via_pydicom_frame_api(ds, frame_num, n_frames)
    if frame is not None:
        return frame

    if arr is None:
        return None

    fi = resolve_frame_index(frame_num, n_frames)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 4:
        return grayscale_frame_2d(arr[fi])

    if arr.ndim == 3:
        if rows and cols:
            if arr.shape[0] == rows and arr.shape[1] == cols:
                return grayscale_frame_2d(arr[:, :, fi])
            if arr.shape[1] == rows and arr.shape[2] == cols:
                return grayscale_frame_2d(arr[fi])
            if arr.shape[0] == cols and arr.shape[1] == rows:
                return grayscale_frame_2d(arr[:, :, fi].T)
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] > 4 and arr.shape[1] > 4:
            return grayscale_frame_2d(arr)
        if arr.shape[0] > 1:
            return grayscale_frame_2d(arr[fi])

    return None


def array_to_image(
    ds: Dataset,
    arr: np.ndarray,
    frame_num: str,
    rows_hint: Optional[int] = None,
    cols_hint: Optional[int] = None,
) -> Optional[Image.Image]:
    frame = extract_frame_pixels(ds, arr, frame_num, rows_hint, cols_hint)
    if frame is None or frame.ndim != 2:
        return None
    gray = normalize_to_uint8(frame)
    return Image.fromarray(gray, mode="L").convert("RGB")


def describe_extract_failure(
    ds: Dataset,
    arr: Optional[np.ndarray],
    frame_num: str,
    rows_hint: Optional[int] = None,
    cols_hint: Optional[int] = None,
) -> str:
    rows, cols = dicom_rows_cols(ds, rows_hint, cols_hint)
    n_frames = count_frames(ds, arr, rows, cols) if arr is not None else 0
    shape = getattr(arr, "shape", None)
    return (
        f"frame={frame_num} resolved_idx={resolve_frame_index(frame_num, max(n_frames, 1))} "
        f"shape={shape} Rows={rows} Cols={cols} NumberOfFrames={getattr(ds, 'NumberOfFrames', '?')}"
    )


def blank_cell(size: Tuple[int, int] = DEFAULT_CELL_SIZE) -> Image.Image:
    return Image.new("RGB", size, color=GRID_BG)


def compose_grid(cells: List[Image.Image], cols: int = 4, pad: int = 6) -> Image.Image:
    cell_w = max((c.width for c in cells), default=DEFAULT_CELL_SIZE[0])
    cell_h = max((c.height for c in cells), default=DEFAULT_CELL_SIZE[1])
    rows = (len(cells) + cols - 1) // cols

    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=GRID_BG)

    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        if cell.size != (cell_w, cell_h):
            cell = cell.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        canvas.paste(cell, (x, y))

    return canvas


def label_image(img: Image.Image, title: str) -> Image.Image:
    bar_h = 22
    out = Image.new("RGB", (img.width, img.height + bar_h), color=GRID_BG)
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), title, fill=(203, 213, 225))
    return out


def process_case_dir(
    case_dir: Path,
    out_case_dir: Path,
    rel_prefix: Path,
    verbose: bool = False,
) -> dict:
    stats = {
        "clips": 0,
        "pngs": 0,
        "grids": 0,
        "skipped": 0,
        "no_annotation": 0,
        "dicom_errors": 0,
        "errors": [],
    }
    clip_images: Dict[str, Dict[str, Image.Image]] = {}
    zone_by_hash: Dict[str, str] = {}
    cell_size: Optional[Tuple[int, int]] = None

    json_files = sorted(case_dir.glob("*.manual_lus_score.json"))
    if not json_files:
        return stats

    dicom_cache: Dict[str, Tuple[Optional[Dataset], Optional[np.ndarray]]] = {}

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        max_frames = max_frames_from_json(data)
        if max_frames is None:
            stats["no_annotation"] += 1
            continue

        file_hash = data.get("file_hash") or json_path.name.split(".")[0]
        dcm_path = case_dir / f"{file_hash}.dcm"
        if not dcm_path.is_file():
            stats["skipped"] += 1
            if verbose:
                stats["errors"].append(f"missing DICOM for {file_hash}")
            continue

        dcm_key = str(dcm_path)
        if dcm_key not in dicom_cache:
            ds, arr, err = load_dicom(dcm_path)
            if err:
                stats["dicom_errors"] += 1
                if len(stats["errors"]) < 3:
                    stats["errors"].append(err)
            dicom_cache[dcm_key] = (ds, arr)

        ds, arr = dicom_cache[dcm_key]
        if ds is None or arr is None:
            stats["skipped"] += 1
            continue

        frames_needed = {max_frames[m] for m in METRICS if max_frames[m] is not None}
        rows_hint = data.get("image_size_rows")
        cols_hint = data.get("image_size_cols")
        frame_images = {
            fn: array_to_image(ds, arr, fn, rows_hint, cols_hint) for fn in frames_needed
        }
        if not any(frame_images.values()):
            stats["skipped"] += 1
            if len(stats["errors"]) < 5:
                fn = sorted(frames_needed)[0]
                stats["errors"].append(
                    "frame extract failed "
                    + describe_extract_failure(ds, arr, fn, rows_hint, cols_hint)
                )
            continue

        zone_by_hash[file_hash] = data.get("lung_zone") or ""
        clip_images[file_hash] = {}
        stats["clips"] += 1
        out_case_dir.mkdir(parents=True, exist_ok=True)

        for metric in METRICS:
            frame_num = max_frames[metric]
            if frame_num is None:
                continue
            img = frame_images.get(frame_num)
            if img is None:
                continue

            if cell_size is None:
                cell_size = img.size

            out_path = out_case_dir / f"{file_hash}_{metric}.png"
            img.save(out_path, format="PNG", compress_level=3)
            clip_images[file_hash][metric] = img
            stats["pngs"] += 1

    if not clip_images:
        return stats

    zone_to_hash = {zone: fh for fh, zone in zone_by_hash.items() if zone}
    blank = blank_cell(cell_size or DEFAULT_CELL_SIZE)

    for metric in METRICS:
        cells = []
        for zone in LUNG_ZONE_ORDER:
            fh = zone_to_hash.get(zone)
            img = clip_images.get(fh, {}).get(metric) if fh else None
            if img is not None:
                cells.append(label_image(img, zone))
            else:
                cells.append(blank.copy())

        grid_path = out_case_dir / f"grid_{metric}.png"
        compose_grid(cells, cols=4, pad=6).save(grid_path, format="PNG", compress_level=3)
        stats["grids"] += 1

    return stats


def iter_case_dirs(assignments_root: Path, phase_filter: Optional[str]):
    for phase_dir in sorted(assignments_root.iterdir()):
        if not phase_dir.is_dir():
            continue
        if phase_filter and phase_dir.name != phase_filter:
            continue
        for annotator_dir in sorted(phase_dir.iterdir()):
            if not annotator_dir.is_dir():
                continue
            for case_dir in sorted(annotator_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                if not any(case_dir.glob("*.manual_lus_score.json")):
                    continue
                yield case_dir, case_dir.relative_to(assignments_root)


def _process_case_job(args: Tuple[Path, Path, Path, bool]) -> Tuple[str, dict]:
    case_dir, out_case_dir, rel_prefix, verbose = args
    stats = process_case_dir(case_dir, out_case_dir, rel_prefix, verbose=verbose)
    return str(rel_prefix), stats


def iter_manual_lus_jsons(assignments_root: Path, phase_filter: Optional[str]):
    """Yield (json_path, case_dir, rel_prefix) for every manual_lus_score JSON."""
    for case_dir, rel_prefix in iter_case_dirs(assignments_root, phase_filter):
        for json_path in sorted(case_dir.glob("*.manual_lus_score.json")):
            yield json_path, case_dir, rel_prefix


def default_test_dcm(assignments_root: Path) -> Path:
    return assignments_root / DEFAULT_TEST_DCM_REL


def resolve_test_clip(
    test_dcm: Path, assignments_root: Path
) -> Tuple[Path, Path, Path]:
    """Return (json_path, dcm_path, rel_prefix) for a test DICOM file."""
    dcm_path = test_dcm.resolve()
    if not dcm_path.is_file():
        raise FileNotFoundError(f"Test DICOM not found: {dcm_path}")

    case_dir = dcm_path.parent
    file_hash = dcm_path.stem
    json_candidates = sorted(case_dir.glob(f"{file_hash}*.manual_lus_score.json"))
    if not json_candidates:
        raise FileNotFoundError(
            f"No .manual_lus_score.json for '{file_hash}' in {case_dir}"
        )

    json_path = json_candidates[0]
    for candidate in json_candidates:
        if "andrew" in candidate.name:
            json_path = candidate
            break

    try:
        rel_prefix = case_dir.relative_to(assignments_root.resolve())
    except ValueError:
        rel_prefix = Path(case_dir.name)

    return json_path, dcm_path, rel_prefix


def run_test_one_clip(
    json_path: Path,
    dcm_path: Path,
    rel_prefix: Path,
    output_root: Path,
) -> int:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    max_frames = max_frames_from_json(data)
    if max_frames is None:
        print(f"No frame_annotations in {json_path}", file=sys.stderr)
        return 1

    file_hash = data.get("file_hash") or dcm_path.stem
    metric = next((m for m in METRICS if max_frames[m] is not None), None)
    if metric is None:
        print("No metric frames found in annotations", file=sys.stderr)
        return 1

    frame_num = max_frames[metric]
    zone = data.get("lung_zone", "")
    print(f"\nClip: {rel_prefix / json_path.name}")
    print(f"  DICOM: {dcm_path}")
    print(f"  {metric} max → frame {frame_num}  |  zone: {zone}")
    print(f"  all max frames: {max_frames}")

    ds, arr, err = load_dicom(dcm_path)
    if err or ds is None or arr is None:
        print(f"  DICOM read failed: {err}", file=sys.stderr)
        return 1

    print(
        f"  pixel shape={arr.shape}  Rows={getattr(ds, 'Rows', '?')}  "
        f"Cols={getattr(ds, 'Columns', '?')}  NumberOfFrames={getattr(ds, 'NumberOfFrames', '?')}"
    )
    rows_hint = data.get("image_size_rows")
    cols_hint = data.get("image_size_cols")
    if rows_hint or cols_hint:
        print(f"  JSON image_size: {rows_hint}×{cols_hint}")

    img = array_to_image(ds, arr, frame_num, rows_hint, cols_hint)
    if img is None:
        print("  frame extract failed:", file=sys.stderr)
        print(f"    {describe_extract_failure(ds, arr, frame_num, rows_hint, cols_hint)}", file=sys.stderr)
        return 1

    test_dir = output_root / "_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_path = test_dir / f"{file_hash}_{metric}.png"
    img.save(out_path, format="PNG", compress_level=3)

    title = f"{file_hash} · {metric} frame {frame_num} · {zone}"
    print(f"\nSaved: {out_path.resolve()}")
    print(f"  {img.size[0]}×{img.size[1]} px — opening matplotlib window …")
    show_test_image(img, title)
    return 0


def show_test_image(img: Image.Image, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — PNG saved but not displayed.", file=sys.stderr)
        print("  pip install matplotlib", file=sys.stderr)
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_test_extract(
    assignments_root: Path,
    output_root: Path,
    test_dcm: Path,
) -> int:
    """Extract one frame from test_dcm, save PNG, display with matplotlib."""
    print("Test mode: one clip → one PNG → matplotlib preview")
    print(f"  test DICOM: {test_dcm}")

    try:
        json_path, dcm_path, rel_prefix = resolve_test_clip(test_dcm, assignments_root)
    except FileNotFoundError as exc:
        print(f"Test setup failed: {exc}", file=sys.stderr)
        return 1

    return run_test_one_clip(json_path, dcm_path, rel_prefix, output_root)


def _print_case_stats(rel_prefix: str, stats: dict, verbose: bool) -> None:
    print(
        f"  {rel_prefix}: clips={stats['clips']} pngs={stats['pngs']} grids={stats['grids']} "
        f"skipped={stats['skipped']} no_annotation={stats['no_annotation']} "
        f"dicom_errors={stats['dicom_errors']}"
    )
    if stats.get("errors") and (verbose or stats["pngs"] == 0):
        for err in stats["errors"]:
            print(f"    ! {err}")


def main():
    parser = argparse.ArgumentParser(description="Extract max-frame PNGs from LUS annotations")
    parser.add_argument("--assignments-root", type=Path, default=Path("completed-assignments-all"))
    parser.add_argument("--output-root", type=Path, default=Path("pngs"))
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-case skip reasons")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test one clip: save PNG under <output-root>/_test/ and show in matplotlib",
    )
    parser.add_argument(
        "--test-dcm",
        type=Path,
        default=None,
        help=f"DICOM file for --test (default: {DEFAULT_TEST_DCM_REL})",
    )
    args = parser.parse_args()

    assignments_root = args.assignments_root.resolve()
    output_root = args.output_root.resolve()

    if not assignments_root.is_dir():
        print(f"Assignments root not found: {assignments_root}", file=sys.stderr)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output PNG root: {output_root}")

    try:
        import pylibjpeg  # noqa: F401
        print("DICOM: pylibjpeg available (JPEG-compressed DICOM supported)")
    except ImportError:
        print(
            "WARNING: pylibjpeg not installed — compressed DICOM may fail. "
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )

    if args.test:
        test_dcm = (args.test_dcm or default_test_dcm(assignments_root)).resolve()
        sys.exit(run_test_extract(assignments_root, output_root, test_dcm))

    jobs = []
    for case_dir, rel_prefix in iter_case_dirs(assignments_root, args.phase):
        jobs.append((case_dir, output_root / rel_prefix, rel_prefix, args.verbose))

    if args.limit:
        jobs = jobs[: args.limit]

    if not jobs:
        print("No case directories found.")
        return

    totals = {
        "cases": 0,
        "clips": 0,
        "pngs": 0,
        "grids": 0,
        "skipped": 0,
        "no_annotation": 0,
        "dicom_errors": 0,
    }
    sample_errors: List[str] = []

    if args.jobs <= 1:
        for case_dir, out_case_dir, rel_prefix, verbose in jobs:
            print(f"Processing {rel_prefix} …")
            stats = process_case_dir(case_dir, out_case_dir, rel_prefix, verbose=verbose)
            totals["cases"] += 1
            for k in totals:
                if k != "cases":
                    totals[k] += stats.get(k, 0)
            for err in stats.get("errors", []):
                if len(sample_errors) < 10:
                    sample_errors.append(f"{rel_prefix}: {err}")
            _print_case_stats(str(rel_prefix), stats, verbose)
    else:
        print(f"Processing {len(jobs)} cases with {args.jobs} workers …")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_process_case_job, item): item for item in jobs}
            done = 0
            for fut in as_completed(futures):
                rel_prefix, stats = fut.result()
                done += 1
                totals["cases"] += 1
                for k in totals:
                    if k != "cases":
                        totals[k] += stats.get(k, 0)
                for err in stats.get("errors", []):
                    if len(sample_errors) < 10:
                        sample_errors.append(f"{rel_prefix}: {err}")
                if args.verbose or stats["pngs"] > 0 or stats["dicom_errors"] > 0 or stats["errors"] or done % 25 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}]", end=" ")
                    _print_case_stats(rel_prefix, stats, verbose)

    print(
        f"\nDone. cases={totals['cases']} clips={totals['clips']} "
        f"pngs={totals['pngs']} grids={totals['grids']} skipped={totals['skipped']} "
        f"no_annotation={totals['no_annotation']} dicom_errors={totals['dicom_errors']}"
    )

    if totals["pngs"] == 0:
        print("\nNo PNGs were written. Common causes:", file=sys.stderr)
        print("  • pip install -r requirements.txt  (needs pylibjpeg for compressed DICOM)", file=sys.stderr)
        print("  • Clips have empty frame_annotations (see no_annotation count)", file=sys.stderr)
        print("  • Missing .dcm next to .manual_lus_score.json (see skipped count)", file=sys.stderr)
        print("  • Re-run with --verbose to see DICOM / frame-extract errors per case", file=sys.stderr)
        if sample_errors:
            print("\nSample errors:", file=sys.stderr)
            for err in sample_errors:
                print(f"  {err}", file=sys.stderr)
    else:
        print(f"PNG files written under: {output_root}")


if __name__ == "__main__":
    main()
