#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIGURATION (modify if needed) ======
# Root directory: should contain 07_dataset_numisheet_tp_8.zip and tool_radii2_* folders
ROOT_DIR="${1:-$(pwd)}"

# Converter script paths (change to your actual paths)
EXCEL_TO_PT_PY="/home/RUS_CIP/st186635/format_transformate/xlsx2pt.py"   # Excel -> .pt
STEP_TO_PT_PY="/home/RUS_CIP/st186635/format_transformate/step2pt.py"    # STEP -> .pt

# Optional parameters for STEP conversion (leave empty to disable)
STEP_LC="${STEP_LC:-}"                # Example: 1.5
STEP_TARGET_POINTS="${STEP_TARGET_POINTS:-}"   # Example: 80000

# Skip existing .pt files? (default = 1 = skip; set to 0 to overwrite)
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# ==============================================

echo "[Info] ROOT_DIR = ${ROOT_DIR}"
echo "[Info] Using Excel converter: ${EXCEL_TO_PT_PY}"
echo "[Info] Using STEP converter : ${STEP_TO_PT_PY}"

# If the dataset zip exists, extract it (with -n to avoid overwriting existing files)
if [ -f "${ROOT_DIR}/07_dataset_numisheet_tp_8.zip" ]; then
  echo "[Info] Found zip, extracting ..."
  (cd "${ROOT_DIR}" && unzip -n "07_dataset_numisheet_tp_8.zip")
fi

# Construct optional STEP arguments
STEP_OPTS=()
if [ -n "${STEP_LC}" ]; then
  STEP_OPTS+=(--lc "${STEP_LC}")
fi
if [ -n "${STEP_TARGET_POINTS}" ]; then
  STEP_OPTS+=(--target_points "${STEP_TARGET_POINTS}")
fi

# Iterate through all tool_* directories
shopt -s nullglob
for tooldir in "${ROOT_DIR}"/tool_*; do
  [ -d "${tooldir}" ] || continue
  datadir="${tooldir}/Data"
  [ -d "${datadir}" ] || { echo "[Warn] No Data/ in ${tooldir}, skip"; continue; }

  echo "=============================="
  echo "[Info] Processing ${datadir}"
  echo "=============================="

  # 1) Process Excel files
  for excel in "${datadir}"/*.xlsx "${datadir}"/*.xls; do
    [ -e "${excel}" ] || continue
    out="${excel%.*}.pt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
      echo "[Skip][Excel] ${out} already exists"
      continue
    fi
    echo "[Run ][Excel] ${excel} -> ${out}"
    # Your xlsx2pt.py automatically detects column names (x (mm), y (mm), z (mm), shell thickness (mm)).
    # If you want to specify manually, append parameters such as:
    #   --xyz-cols "x (mm),y (mm),z (mm)" --thick-col "shell thickness (mm)"
    python "${EXCEL_TO_PT_PY}" "${excel}" "${out}"
  done

  # 2) Process STEP files
  for stepf in "${datadir}"/*.step; do
    [ -e "${stepf}" ] || continue
    out="${stepf%.*}.pt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
      echo "[Skip][STEP ] ${out} already exists"
      continue
    fi
    echo "[Run ][STEP ] ${stepf} -> ${out}"
    # If your step2pt.py requires additional parameters (e.g., voxel size, output name), append them here.
    python "${STEP_TO_PT_PY}" "${stepf}" "${out}" "${STEP_OPTS[@]}"
  done
done

echo "[Done] All conversions finished."
