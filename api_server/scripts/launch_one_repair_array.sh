#!/usr/bin/env bash
set -uo pipefail
root="${PIPELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
exclude_args=()
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
  exclude_args=(--exclude="$SLURM_EXCLUDE_NODES")
fi
list="$root/output_v2/asset_scale_repair.list"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  count=$(wc -l < "$list")
  last=1000
  (( count - 1 < last )) && last=$((count - 1))
  sbatch --parsable --job-name=asset-scale-repair --partition=main --qos=low \
    --array="0-${last}%32" --export="ALL,REPAIR_OFFSET=0" --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=8G "${exclude_args[@]}" --time=2-00:00:00 \
    --output="$root/output_v2/asset-scale-repair-%A_%a.log" \
    --wrap="bash '$root/scripts/repair_existing_assets_from_source.sh' run"
  exit $?
fi
exec env -u SLURM_JOB_ID -u SLURM_ARRAY_TASK_ID bash "$root/scripts/repair_existing_assets_from_source.sh"
