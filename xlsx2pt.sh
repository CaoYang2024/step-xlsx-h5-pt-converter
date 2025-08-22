#!/usr/bin/env bash
set -euo pipefail

# ====== 配置区（按需修改） ======
# 根目录：包含 tool_* 文件夹
ROOT_DIR="/mnt/data/hsh/raw/"

# 转换脚本路径（改成你的实际路径）
EXCEL_TO_PT_PY="/home/RUS_CIP/st186635/format_transformate/xlsx2pt.py"
# ===============================

echo "[Info] ROOT_DIR = ${ROOT_DIR}"
echo "[Info] Using Excel converter: ${EXCEL_TO_PT_PY}"

shopt -s nullglob
for tooldir in "${ROOT_DIR}"/tool_*; do
  [ -d "${tooldir}" ] || continue
  datadir="${tooldir}/Data"
  [ -d "${datadir}" ] || { echo "[Warn] No Data/ in ${tooldir}, skip"; continue; }

  echo "=============================="
  echo "[Info] Processing ${datadir}"
  echo "=============================="

  for excel in "${datadir}"/*.xlsx "${datadir}"/*.xls; do
    [ -e "${excel}" ] || continue
    out="${excel%.*}.pt"
    echo "[Run ][Excel] ${excel} -> ${out}"
    python "${EXCEL_TO_PT_PY}" "${excel}" "${out}"
  done
done

echo "[Done] All Excel conversions finished."
