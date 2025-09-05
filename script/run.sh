#!/usr/bin/env bash
set -euo pipefail

# ====== 配置区（按需修改） ======
# 根目录：包含 07_dataset_numisheet_tp_8.zip 和那些 tool_radii2_* 文件夹
ROOT_DIR="${1:-$(pwd)}"

# 转换脚本路径（请改成你的实际路径）
EXCEL_TO_PT_PY="/home/RUS_CIP/st186635/format_transformate/xlsx2pt.py"       # 你刚写的 xlsx -> pt
STEP_TO_PT_PY="/home/RUS_CIP/st186635/format_transformate/step2pt.py"     # 你的 step -> pt 脚本

# STEP 转换的可选参数（留空就不传；比如你之前用过 --lc 和 --target_points）
STEP_LC="${STEP_LC:-}"                # 例如：1.5
STEP_TARGET_POINTS="${STEP_TARGET_POINTS:-}"   # 例如：80000

# 是否跳过已存在的 .pt（默认 1 表示跳过；设为 0 代表覆盖重跑）
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# ===============================

echo "[Info] ROOT_DIR = ${ROOT_DIR}"
echo "[Info] Using Excel converter: ${EXCEL_TO_PT_PY}"
echo "[Info] Using STEP converter : ${STEP_TO_PT_PY}"

# 如果有压缩包，先解压（-n 不覆盖已存在文件）
if [ -f "${ROOT_DIR}/07_dataset_numisheet_tp_8.zip" ]; then
  echo "[Info] Found zip, extracting ..."
  (cd "${ROOT_DIR}" && unzip -n "07_dataset_numisheet_tp_8.zip")
fi

# 构造 STEP 可选参数
STEP_OPTS=()
if [ -n "${STEP_LC}" ]; then
  STEP_OPTS+=(--lc "${STEP_LC}")
fi
if [ -n "${STEP_TARGET_POINTS}" ]; then
  STEP_OPTS+=(--target_points "${STEP_TARGET_POINTS}")
fi

# 遍历所有 tool_* 目录
shopt -s nullglob
for tooldir in "${ROOT_DIR}"/tool_*; do
  [ -d "${tooldir}" ] || continue
  datadir="${tooldir}/Data"
  [ -d "${datadir}" ] || { echo "[Warn] No Data/ in ${tooldir}, skip"; continue; }

  echo "=============================="
  echo "[Info] Processing ${datadir}"
  echo "=============================="

  # 1) 处理 Excel
  for excel in "${datadir}"/*.xlsx "${datadir}"/*.xls; do
    [ -e "${excel}" ] || continue
    out="${excel%.*}.pt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
      echo "[Skip][Excel] ${out} already exists"
      continue
    fi
    echo "[Run ][Excel] ${excel} -> ${out}"
    # 你的 main.py 已自动识别列名（x (mm), y (mm), z (mm), shell thickness (mm)）
    # 若想手动指定列名，可在下面命令末尾加：
    #   --xyz-cols "x (mm),y (mm),z (mm)" --thick-col "shell thickness (mm)"
    python "${EXCEL_TO_PT_PY}" "${excel}" "${out}"
  done

  # 2) 处理 STEP
  for stepf in "${datadir}"/*.step; do
    [ -e "${stepf}" ] || continue
    out="${stepf%.*}.pt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
      echo "[Skip][STEP ] ${out} already exists"
      continue
    fi
    echo "[Run ][STEP ] ${stepf} -> ${out}"
    # 你的 step2pt.py 如果需要其他参数（比如输出名、体素大小等），在此追加
    python "${STEP_TO_PT_PY}" "${stepf}" "${out}" "${STEP_OPTS[@]}"
  done
done

echo "[Done] All conversions finished."
