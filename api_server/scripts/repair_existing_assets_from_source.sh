#!/usr/bin/env bash
set -uo pipefail

root="${PIPELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
exclude_args=()
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
  exclude_args=(--exclude="$SLURM_EXCLUDE_NODES")
fi
asset_root="$root/output_v2/asset"
source_root="$root/output_v2/blender"
list="$root/output_v2/asset_scale_repair.list"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' "$asset_root"/*/*/0 > "$list"

  count="$(wc -l < "$list")"
  if (( count == 0 )); then
    echo "no repair candidates"
    exit 0
  fi
  jobs=()
  chunk_size=1001
  for ((offset=0; offset<count; offset+=chunk_size)); do
    end=$((offset + chunk_size - 1))
    (( end >= count )) && end=$((count - 1))
    job="$(sbatch --parsable --job-name=asset-scale-repair --partition=main --qos=low \
      --array="0-$((end - offset))%64" --export="ALL,REPAIR_OFFSET=$offset" \
      --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
      "${exclude_args[@]}" --time=2-00:00:00 \
      --output="$root/output_v2/asset-scale-repair-%A_%a.log" \
      --wrap="bash '$0' run")"
    jobs+=("$job")
  done
  dependency="afterok:${jobs[0]}"
  for job in "${jobs[@]:1}"; do dependency="$dependency:$job"; done
  annotation_job="$(sbatch --parsable --dependency="$dependency" --job-name=asset-annotation-refresh \
    --partition=main --qos=low --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G \
    "${exclude_args[@]}" --time=2-00:00:00 \
    --output="$root/output_v2/step_5_annotation-refresh-%j.log" \
    --chdir="$root" --wrap="bash scripts/step_5_annotation.sh all 16 0")"
  echo "repair_jobs=${jobs[*]} candidates=$count annotation_job=$annotation_job"
  exit 0
fi

offset="${REPAIR_OFFSET:-0}"
index=$((SLURM_ARRAY_TASK_ID + offset + 1))
dir="$(sed -n "${index}p" "$list")"
[[ -n "$dir" ]] || exit 0
sample="$(basename "$(dirname "$dir")")"
category="$(basename "$(dirname "$(dirname "$dir")")")"
source="$source_root/$category/$sample/0/scene.blend"
runner="$dir/extract_and_render.py"
log="$dir/_debug/scale_repair.log"
mkdir -p "$dir/_debug"

timeout 3600 "${BLENDER_BIN:-blender}" -b --threads 1 \
  --python "$runner" -- "$source" "$dir" > "$log" 2>&1
code=$?
if (( code == 0 )) && [[ -s "$dir/${sample}_only_color.blend" && -s "$dir/${sample}_asset_preview_color.png" ]]; then
  python - "$dir/status.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text()) if p.exists() else {}
d["status"] = "pass"
d["scale_repaired_from_source"] = True
d["annotation"] = {"status": "pending", "reason": "refreshed after source-scene asset repair"}
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
PY
  rm -rf "$root/output_v2/annotation/$category/$sample/0"
else
  echo "repair_failed exit=$code category=$category sample=$sample" >> "$log"
fi
exit 0
