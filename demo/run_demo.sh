#!/usr/bin/env bash
# End-to-end THiRST run on the synthetic exemplar dataset (stages 0-4 + object/spot matching).
#
#   bash demo/run_demo.sh [DATA_DIR] [--real]
#
# Without --real the XFeat/ANTs registration is replaced by the ground-truth transforms written by
# demo/make_synthetic_data.py (no GPU, no ANTs needed); default DATA_DIR: demo_data.
# With --real, XFeat (torch.hub) and ANTs estimate the registration; default DATA_DIR: demo_data_real.
#
# Registration steps keep results that already exist, so each mode needs its own DATA_DIR: the mode
# is recorded in DATA_DIR/.demo_mode and the script refuses to reuse a directory of the other mode
# (a --real run must never pick up the simulated ground-truth transforms).
#
# The complete output is appended to DATA_DIR/demo_run.log; DATA_DIR/demo_environment.txt records the
# command, the repository commit and the versions of Python, the libraries, torch, XFeat and ANTs.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repository root

REAL=""
DATA=""
for arg in "$@"; do
  case "$arg" in
    --real) REAL="--real" ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option: $arg (usage: bash demo/run_demo.sh [DATA_DIR] [--real])" >&2; exit 2 ;;
    *) DATA="$arg" ;;
  esac
done
if [ -n "$REAL" ]; then MODE=real; else MODE=simulated; fi
if [ -z "$DATA" ]; then
  if [ -n "$REAL" ]; then DATA=demo_data_real; else DATA=demo_data; fi
fi
CFG="$DATA/demo.yaml"
MARK="$DATA/.demo_mode"
S=scripts
export MPLBACKEND=Agg

step() { echo; echo "=== $*"; }


if [ -f "$CFG" ] && [ ! -f "$MARK" ]; then
  echo "error: $DATA has no $MARK (created by an older version of this script?)." >&2
  echo "       Use a new DATA_DIR so that no earlier registration result is reused." >&2
  exit 1
fi
if [ -f "$CFG" ] && [ "$(cat "$MARK")" != "$MODE" ]; then
  echo "error: $DATA holds a $(cat "$MARK") run; a $MODE run would reuse its registration results." >&2
  echo "       Use another directory, e.g.: bash demo/run_demo.sh ${DATA}_${MODE} $REAL" >&2
  exit 1
fi

mkdir -p "$DATA"
exec > >(tee -a "$DATA/demo_run.log") 2>&1      # keep the complete log next to the data
ENV_FILE="$DATA/demo_environment.txt"
{
  echo "date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "command: bash demo/run_demo.sh $*"
  echo "mode: $MODE"
  echo "THiRST commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo ' (with uncommitted changes)')"
  python - <<'PY'
import importlib, platform
print("python:", platform.python_version(), platform.platform())
for name in ("numpy", "pandas", "scipy", "cv2", "shapely", "SimpleITK", "skimage", "openslide", "joblib", "torch"):
    try:
        mod = importlib.import_module(name)
        version = mod.Version_VersionString() if name == "SimpleITK" else getattr(mod, "__version__", "?")
    except Exception:
        version = "not installed"
    print(f"{name}: {version}")
PY
  if [ "$MODE" = real ]; then
    ANTS_REG="$(command -v antsRegistration || echo "${ANTSPATH:+$ANTSPATH/}antsRegistration")"
    echo "ANTs: $("$ANTS_REG" --version 2>/dev/null | head -n 1 || echo unknown) ($ANTS_REG)"
  fi
} > "$ENV_FILE"

if [ ! -f "$CFG" ]; then
  step "create synthetic data in $DATA ($MODE registration)"
  python demo/make_synthetic_data.py create --out "$DATA"
  echo "$MODE" > "$MARK"
fi

step "0 | ST unit centres + section images, ST unit polygons"
python $S/0_preprocessing/export_st_centers.py   --config "$CFG" --save-png
python $S/0_preprocessing/export_st_contours.py  --config "$CFG"

register() {   # $1 = global | patch
  if [ "$MODE" = real ]; then
    python $S/1_registration/xfeat_affine.py  --config "$CFG" --level "$1"
    python $S/1_registration/affine_to_itk.py --config "$CFG" --level "$1"
    python $S/1_registration/ants_affine.py   --config "$CFG" --level "$1"
  else
    python demo/make_synthetic_data.py simulate-registration --config "$CFG" --level "$1"
    python $S/1_registration/affine_to_itk.py --config "$CFG" --level "$1"
  fi
  python $S/1_registration/warp_to_reference.py --config "$CFG" --level "$1"
}

step "1 | global registration to the middle section ($MODE)"
register global

step "2 | global coordinate transfer"
python $S/2_coordinate_transfer/global_spot_centers.py  --config "$CFG"
python $S/2_coordinate_transfer/global_spot_contours.py --config "$CFG"
python $S/2_coordinate_transfer/global_objects.py       --config "$CFG"

step "1 | patch sampling, cropping and patch-level registration ($MODE)"
python $S/1_registration/sample_patch_centers.py --config "$CFG"
python $S/1_registration/crop_patch_stacks.py    --config "$CFG"
register patch

step "2 | patch-refined coordinate transfer"
python $S/2_coordinate_transfer/patch_spot_centers.py  --config "$CFG"
python $S/2_coordinate_transfer/patch_spot_contours.py --config "$CFG"
python $S/2_coordinate_transfer/patch_objects.py       --config "$CFG"

step "3 | ST unit association (V2-V2 and V2-HD IoU)"
python $S/3_st_association/v2_v2_iou.py       --config "$CFG" --registration patch
python $S/3_st_association/v2_hd_iou.py       --config "$CFG" --registration patch
python $S/3_st_association/merge_v2_hd_iou.py --config "$CFG" --registration patch

step "4 | 3D object reconstruction"
python $S/4_object_tracking/track_objects.py      --config "$CFG"
python $S/4_object_tracking/check_track_counts.py --config "$CFG"
python $S/4_object_tracking/cross_section_area.py --config "$CFG" --tracks-subdir Object_IoU_patch_connectglom

step "5 | ST units inside every 3D object"
python $S/5_object_analysis/match_objects_to_spots.py --config "$CFG"
python $S/5_object_analysis/export_track_barcodes.py  --config "$CFG"

step "check against the ground truth"
python demo/check_demo.py --config "$CFG"

if [ "$MODE" = real ]; then           # what XFeat actually ran (code, commit, weights)
  for f in "$DATA"/WSI_png/xfeat_provenance_*.json; do
    { echo; echo "== $(basename "$f")"; cat "$f"; } >> "$ENV_FILE"
  done
fi
echo "log: $DATA/demo_run.log; environment: $ENV_FILE"
