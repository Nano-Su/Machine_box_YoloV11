#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS])


def infer_label_dir(image_dir: Path) -> Path:
    # Roboflow default: split/images -> split/labels
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir.parent / "labels"


def read_label_rows(p: Path):
    # rows: (class_id, w, h) only (normalized)
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    rows = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            w = float(parts[3])
            h = float(parts[4])
            rows.append((cid, w, h))
        except ValueError:
            continue
    return rows


def quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--root", default=".", help="Root used to resolve relative paths in data.yaml")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))

    names = cfg.get("names") or []
    nc = int(cfg.get("nc") or len(names))
    if names and len(names) != nc:
        nc = len(names)

    splits_cfg = [("train", cfg.get("train")), ("val", cfg.get("val")), ("test", cfg.get("test"))]

    class_instance = [0] * nc
    class_images = [0] * nc
    areas: List[float] = []

    print("=== Split Summary ===")
    for split_name, rel_path in splits_cfg:
        if not rel_path:
            continue
        img_dir = (root / str(rel_path)).resolve()
        lbl_dir = infer_label_dir(img_dir)

        imgs = list_images(img_dir)
        labeled_images = 0
        missing_label_images = 0
        empty_label_images = 0
        total_boxes = 0

        for img in imgs:
            label_path = lbl_dir / f"{img.stem}.txt"
            rows = read_label_rows(label_path)

            if not label_path.exists():
                missing_label_images += 1
                continue

            labeled_images += 1
            if not rows:
                empty_label_images += 1
                continue

            total_boxes += len(rows)
            present = set()

            for cid, w, h in rows:
                if 0 <= cid < nc:
                    class_instance[cid] += 1
                    present.add(cid)
                areas.append(max(0.0, min(1.0, w)) * max(0.0, min(1.0, h)))

            for cid in present:
                class_images[cid] += 1

        print(f"{split_name}:")
        print(f"  image_dir: {img_dir}")
        print(f"  label_dir: {lbl_dir}")
        print(f"  images: {len(imgs)}")
        print(f"  labeled_images: {labeled_images}")
        print(f"  missing_label_images: {missing_label_images}")
        print(f"  empty_label_images: {empty_label_images}")
        print(f"  total_boxes: {total_boxes}")

    print("\n=== Class Summary (instances / images) ===")
    for i in range(nc):
        cname = names[i] if i < len(names) else str(i)
        print(f"{i:02d} {cname:>6}: {class_instance[i]:6d} / {class_images[i]:6d}")

    print("\n=== BBox Relative Area Stats (w*h normalized) ===")
    print(f"count: {len(areas)}")
    print(f"min:   {quantile(areas, 0.0):.6f}")
    print(f"p10:   {quantile(areas, 0.1):.6f}")
    print(f"p25:   {quantile(areas, 0.25):.6f}")
    print(f"p50:   {quantile(areas, 0.5):.6f}")
    print(f"p75:   {quantile(areas, 0.75):.6f}")
    print(f"p90:   {quantile(areas, 0.9):.6f}")
    print(f"max:   {quantile(areas, 1.0):.6f}")


if __name__ == "__main__":
    main()