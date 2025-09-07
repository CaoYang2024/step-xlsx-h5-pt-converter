#!/usr/bin/env bash
set -euo pipefail

# Root directory, e.g. pass /mnt/data/hsh/raw (which contains tool_*)
ROOT_DIR="${1:-$(pwd)}"

# Python interpreter and script paths
PYTHON=python
EXCEL_TO_PT="/home/RUS_CIP/st186635/format_transformate/xlsx2pt.py"
PT_TO_PNG="/home/RUS_CIP/st186635/format_transformate/pt2png.py"
PNG_TO_GIF="/home/RUS_CIP/st186635/format_transformate/png2gif.py"

# Iterate through tool_* directories
for tooldir in "${ROOT_DIR}"/tool_*; do
  [ -d "$tooldir/Data" ] || continue
  datadir="$tooldir/Data"

  echo "=== Processing $datadir ==="

  # 1) Excel -> PT (overwrite existing .pt files)
  for excel in "$datadir"/*.xlsx "$datadir"/*.xls; do
    [ -e "$excel" ] || continue
    out="${excel%.*}.pt"
    echo "[Excel->PT] $excel -> $out"
    rm -f "$out"
    $PYTHON "$EXCEL_TO_PT" "$excel" "$out"
  done

  # 2) PT -> PNG (skip 01_/02_/03_ prefixed files)
  for ptf in "$datadir"/*.pt; do
    [ -e "$ptf" ] || continue
    base=$(basename "$ptf")
    if [[ "$base" =~ ^0(1|2|3)_.*\.pt$ ]]; then
      echo "[PT->PNG] Skip $base"
      continue
    fi
    out="${ptf%.*}.png"
    echo "[PT->PNG] $ptf -> $out"
    $PYTHON "$PT_TO_PNG" "$ptf" --png "$out"
  done

  # 3) PNG -> GIF (calls modified Python script, internally reverse-sorted)
  gif_out="$datadir/animation.gif"
  echo "[PNG->GIF] -> $gif_out"
  $PYTHON "$PNG_TO_GIF" "$datadir" --gif "$gif_out" --fps 5 --loop 0
done

echo "=== All done ==="
