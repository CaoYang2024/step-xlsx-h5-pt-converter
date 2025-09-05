#!/usr/bin/env bash
set -euo pipefail

# 用法: bash batch_h5_build.sh <ROOT_DIR> [H5_DIR]
# 例:  bash batch_h5_build.sh /home/RUS_CIP/st186635/format_transformate
# 若不传 H5_DIR，默认写到 <ROOT_DIR>/h5

ROOT="${1:-$PWD}"
H5_DIR="${2:-$ROOT/h5}"
H5FINAL="/home/RUS_CIP/st186635/format_transformate/h5final.py"

mkdir -p "$H5_DIR"
MAP_CSV="$H5_DIR/mapping.csv"
echo "h5_name,h5_path,source_dir,radii2,radii1,delta,cr,height" > "$MAP_CSV"

shopt -s nullglob

# 优先遍历 ROOT/*/Data；如无该结构，退回 ROOT/*（兼容两种目录形态）
DATADIRS=()
if compgen -G "$ROOT"/*/Data > /dev/null; then
  for D in "$ROOT"/*/Data; do [[ -d "$D" ]] && DATADIRS+=("$D"); done
else
  for D in "$ROOT"/*; do [[ -d "$D" ]] && DATADIRS+=("$D"); done
fi

for DATADIR in "${DATADIRS[@]}"; do
  if [[ "$(basename "$DATADIR")" == "Data" ]]; then
    SRCDIR="$(dirname "$DATADIR")"   # Data 的父目录
    BASENAME="$(basename "$SRCDIR")" # 以父目录名做 H5 文件名
    XLSX_DIR="$DATADIR"
  else
    SRCDIR="$DATADIR"
    BASENAME="$(basename "$SRCDIR")"
    if compgen -G "$SRCDIR"/Data/*.xlsx > /dev/null; then
      XLSX_DIR="$SRCDIR/Data"
    elif compgen -G "$SRCDIR"/*.xlsx > /dev/null; then
      XLSX_DIR="$SRCDIR"
    else
      XLSX_DIR=""
    fi
  fi

  OUT="$H5_DIR/$BASENAME.h5"

  # 收集 STEP（优先 Data/，否则父目录）
  declare -A STEP_MAP=( ["die"]="" ["binder"]="" ["punch"]="" )
  step_candidates=( "$SRCDIR"/Data/*.step "$SRCDIR"/*.step )
  for S in "${step_candidates[@]}"; do
    [[ -e "$S" ]] || continue
    name_lc="$(basename "$S" | tr '[:upper:]' '[:lower:]')"
    [[ -z "${STEP_MAP[die]}"    && "$name_lc" == *"matrize"*     ]] && STEP_MAP[die]="$S"
    [[ -z "${STEP_MAP[binder]}" && "$name_lc" == *"niederhalter"* ]] && STEP_MAP[binder]="$S"
    [[ -z "${STEP_MAP[punch]}"  && "$name_lc" == *"stempel"*      ]] && STEP_MAP[punch]="$S"
  done
  # 兜底：按发现顺序把未匹配到的 step 填入空位
  uniq_steps=()
  for S in "${step_candidates[@]}"; do [[ -e "$S" ]] && uniq_steps+=("$S"); done
  mapfile -t uniq_steps < <(printf "%s\n" "${uniq_steps[@]}" | awk '!seen[$0]++')
  idx=0
  for role in die binder punch; do
    if [[ -z "${STEP_MAP[$role]}" && $idx -lt ${#uniq_steps[@]} ]]; then
      STEP_MAP[$role]="${uniq_steps[$idx]}"; ((idx++))
    fi
  done

  # 没有 xlsx、也没有任何 step 就跳过
  if [[ -z "$XLSX_DIR" && -z "${STEP_MAP[die]}" && -z "${STEP_MAP[binder]}" && -z "${STEP_MAP[punch]}" ]]; then
    echo "[Skip] $SRCDIR (no xlsx/step found)"
    continue
  fi

  # 覆盖式：删除旧 H5
  rm -f "$OUT"

  # 组装 h5final.py 命令
  CMD=( python "$H5FINAL" --out "$OUT" --tag-from "$SRCDIR" )
  [[ -n "$XLSX_DIR"          ]] && CMD+=( --xlsx-dir "$XLSX_DIR" )
  [[ -n "${STEP_MAP[die]}"    ]] && CMD+=( --die-step    "${STEP_MAP[die]}" )
  [[ -n "${STEP_MAP[binder]}" ]] && CMD+=( --binder-step "${STEP_MAP[binder]}" )
  [[ -n "${STEP_MAP[punch]}"  ]] && CMD+=( --punch-step  "${STEP_MAP[punch]}" )
  # 你的网格参数
  CMD+=( --quad --size 5.0 )

  echo "[RUN] ${CMD[*]}"
  if ! "${CMD[@]}"; then
    echo "[WARN] h5final.py failed: $SRCDIR" >&2
    continue
  fi

  # —— 从 BASENAME 解析参数（允许负号与小数）
  read -r radii2 radii1 delta cr height < <(python - "$BASENAME" <<'PY'
import re,sys
name=sys.argv[1]
def pick(p): 
    m=re.search(p,name)
    return m.group(1) if m else ""
# 顺序：radii2,radii1,delta,cr,height
vals=[
    pick(r"radii2_([\-0-9]+(?:\.[0-9]+)?)"),
    pick(r"radii1_([\-0-9]+(?:\.[0-9]+)?)"),
    pick(r"delta_([\-0-9]+(?:\.[0-9]+)?)"),
    pick(r"cr_([\-0-9]+(?:\.[0-9]+)?)"),
    pick(r"height_([\-0-9]+(?:\.[0-9]+)?)"),
]
print(*vals)
PY
)

  # —— 把参数写进 H5 的文件属性：单个键 + 合并的 JSON 键 "Parameters"
  python - "$OUT" "$radii2" "$radii1" "$delta" "$cr" "$height" <<'PY'
import sys,json,h5py,math
path, radii2, radii1, delta, cr, height = sys.argv[1:7]
vals = {"radii2":radii2, "radii1":radii1, "delta":delta, "cr":cr, "height":height}
def to_num_or_none(s):
    if s=="":
        return None
    try:
        x=float(s)
        if math.isfinite(x): return x
    except:
        pass
    return s
nums = {k: to_num_or_none(v) for k,v in vals.items()}
with h5py.File(path, "a") as f:
    # 单个标量属性（若有值）
    for k,v in nums.items():
        if v is not None:
            f.attrs[k] = v
    # 合并 JSON 属性
    f.attrs["Parameters"] = json.dumps(nums, ensure_ascii=False)
PY

  # —— 记录到 CSV（追加参数列）
  echo "$BASENAME.h5,$OUT,$SRCDIR,$radii2,$radii1,$delta,$cr,$height" >> "$MAP_CSV"
done

echo "[Done] H5 written to: $H5_DIR"
echo "[Done] CSV: $MAP_CSV"
