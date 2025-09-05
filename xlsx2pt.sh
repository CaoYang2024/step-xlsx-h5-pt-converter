#!/usr/bin/env bash
set -euo pipefail

# ====== 配置区（按需修改） ======
# 根目录：包含 tool_* 文件夹
ROOT_DIR="${ROOT_DIR:-/mnt/data/hsh/raw}"

# 转换脚本路径（改成你的实际路径）
EXCEL_TO_PT_PY="${EXCEL_TO_PT_PY:-/home/RUS_CIP/st186635/format_transformate/xlsx2pt.py}"

# 是否跳过已存在 .pt（1=跳过；0=覆盖重跑）
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Python 解释器（可改为 python3 / venv 里的绝对路径）
PY="${PY:-python3}"
# ===============================

echo "[Info] ROOT_DIR          = ${ROOT_DIR}"
echo "[Info] Excel converter   = ${EXCEL_TO_PT_PY}"
echo "[Info] Python executable = ${PY}"
echo "[Info] SKIP_EXISTING     = ${SKIP_EXISTING}"

# 基本存在性检查
if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "[Error] ROOT_DIR not found: ${ROOT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${EXCEL_TO_PT_PY}" ]]; then
  echo "[Error] Excel converter not found: ${EXCEL_TO_PT_PY}" >&2
  exit 1
fi
if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "[Error] Python not found: ${PY}" >&2
  exit 1
fi

shopt -s nullglob

# 遍历 tool_* 目录
for tooldir in "${ROOT_DIR}"/tool_*; do
  [[ -d "${tooldir}" ]] || continue
  datadir="${tooldir}/Data"
  if [[ ! -d "${datadir}" ]]; then
    echo "[Warn] No Data/ in ${tooldir}, skip"
    continue
  fi

  echo "=============================="
  echo "[Info] Processing ${datadir}"
  echo "=============================="

  # 同时匹配 .xlsx/.xls（大小写都支持），并且排除临时 Excel（~$ 开头）
  # 使用数组，避免通配符在没有匹配时留下字面量
  files=()
  while IFS= read -r -d '' f; do
    files+=("$f")
  done < <(find "${datadir}" -maxdepth 1 -type f \
            \( -iname "*.xlsx" -o -iname "*.xls" \) \
            ! -name "~$*" -print0)

  if (( ${#files[@]} == 0 )); then
    echo "[Info] No Excel files in ${datadir}, skip"
    continue
  fi

  # 逐个转换
  for excel in "${files[@]}"; do
    out="${excel%.*}.pt"

    if [[ "${SKIP_EXISTING}" == "1" && -e "${out}" ]]; then
      echo "[Skip][Excel] ${excel} -> ${out} (exists)"
      continue
    fi

    echo "[Run ][Excel] ${excel} -> ${out}"
    # 若你的 xlsx2pt.py 接口是：python xlsx2pt.py <in> <out>
    # 如有额外参数可在此追加
    if ! "${PY}" "${EXCEL_TO_PT_PY}" "${excel}" "${out}"; then
      echo "[Error] Convert failed: ${excel}" >&2
      # 不直接退出，继续处理后续文件；若想失败即停，用 'exit 1'
    fi
  done
done

echo "[Done] All Excel conversions finished."
