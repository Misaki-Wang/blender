#!/usr/bin/env python3
"""Build a deterministic index of complete, QC-passed isolated assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_SUFFIXES = ("_only_color.blend", "_asset_preview_color.png")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def dimensions_from_bbox(path: Path):
    data = read_json(path)
    candidates = data.get("dimensions_m") or data.get("dimensions")
    if isinstance(candidates, list) and len(candidates) == 3:
        try:
            return [float(x) for x in candidates]
        except (TypeError, ValueError):
            pass
    points = data.get("corners") or data.get("bbox")
    if isinstance(points, list) and points:
        xyz = [[], [], []]
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 3:
                for i in range(3):
                    try:
                        xyz[i].append(float(point[i]))
                    except (TypeError, ValueError):
                        pass
        if all(values for values in xyz):
            return [max(values) - min(values) for values in xyz]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rules = read_json(args.categories)
    default = rules.get("default", {})
    rows = []
    for status_path in sorted(args.asset_root.glob("*/*/0/status.json")):
        out = status_path.parent
        status = read_json(status_path)
        if status.get("status") not in {"pass", "passed"}:
            continue
        category = out.parent.parent.name
        sample = out.parent.name
        files = {path.name for path in out.iterdir() if path.is_file()}
        if not all(any(name.endswith(suffix) for name in files) for suffix in REQUIRED_SUFFIXES):
            continue
        rule = dict(default)
        rule.update(rules.get(category, {}))
        dimensions = dimensions_from_bbox(out / "bbox_3d.json")
        rows.append({
            "asset_id": f"{category}/{sample}",
            "category": category,
            "sample": sample,
            "color_blend": str((out / f"{sample}_only_color.blend").relative_to(args.asset_root.parent)),
            "asset_type": rule.get("asset_type", "rigid"),
            "support_type": rule.get("support_type", "floor"),
            "scale_band": rule.get("scale_band", "medium"),
            "scene_tags": rule.get("scene_tags", []),
            "dimensions_m": dimensions,
            "origin_policy": "bottom_center",
            "front_axis": "+Y",
            "source_status": status.get("status"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"indexed": len(rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
