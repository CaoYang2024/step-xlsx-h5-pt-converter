#!/usr/bin/env bash
set -euo pipefail

# 根目录，传入 /mnt/data/hsh/raw（里面有 tool_*）
ROOT_DIR="${1:-$(pwd)}"

# Python 和脚本路径
PYTHON=python
EXCEL_TO_PT="/home/RUS_CIP/st186635/format_transformate/xlsx2pt.py"
PT_TO_PNG="/home/RUS_CIP/st186635/format_transformate/pt2png.py"
PNG_TO_GIF="/home/RUS_CIP/st186635/format_transformate/png2gif.py"

for tooldir in "${ROOT_DIR}"/tool_*; do
  [ -d "$tooldir/Data" ] || continue
  datadir="$tooldir/Data"

  echo "=== Processing $datadir ==="

  # 1) Excel -> PT（覆盖旧文件）
  for excel in "$datadir"/*.xlsx "$datadir"/*.xls; do
    [ -e "$excel" ] || continue
    out="${excel%.*}.pt"
    echo "[Excel->PT] $excel -> $out"
    rm -f "$out"
    $PYTHON "$EXCEL_TO_PT" "$excel" "$out"
  done

  # 2) PT -> PNG（跳过 01_/02_/03_）
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

  # 3) PNG -> GIF（调用你改过的 py，内部已倒序）
  gif_out="$datadir/animation.gif"
  echo "[PNG->GIF] -> $gif_out"
  $PYTHON "$PNG_TO_GIF" "$datadir" --gif "$gif_out" --fps 5 --loop 0
done

echo "=== All done ==="
