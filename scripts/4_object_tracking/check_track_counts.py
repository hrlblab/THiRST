"""Consistency check of the tracking outputs (per-section object / track counts).

THiRST stage 4 - quality control.

Compares, per WSI section, the number of contours and track IDs in IoU_Object.json (reference frame),
IoU_Object_WSI.json (WSI frame) and the exported QuPath GeoJSONs; rows marked (!) disagree.

Usage
-----
    python scripts/4_object_tracking/check_track_counts.py --config configs/example.yaml [--tracks-suffix _longglom]

Original script(s): Step12_Match_Object_tracking_results_itemcheck.py
"""
import sys
import os
import json
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args  # noqa: E402


def _norm(s): return str(s).strip().lower()

def _load_slice_to_read_from_csv2(csv2_path):
    """Build slice_index -> read_index map from csv2 (patch_contours_before_after.csv)."""
    slice_to_read = {}
    if not os.path.exists(csv2_path):
        print(f"[WARN] csv2 not found: {csv2_path}")
        return slice_to_read
    with open(csv2_path, 'r') as f:
        header = f.readline().strip().split(',')
        idx = {name:i for i,name in enumerate(header)}
        need = ['read_index','slice_idx']
        for k in need:
            if k not in idx:
                raise RuntimeError(f"csv2 missing column '{k}'")
        for line in f:
            s = line.strip()
            if not s: continue
            p = s.split(',')
            try:
                slice_idx = int(p[idx['slice_idx']])
            except:
                # if slice_idx isn't numeric, skip
                continue
            read_idx = _norm(p[idx['read_index']])
            slice_to_read[slice_idx] = read_idx
    return slice_to_read

def _count_from_iou_patch(iou_patch_json, slice_to_read):
    """Counts per read_index and per class from IoU_Object.json (patch space)."""
    per_read = Counter()
    per_read_class = defaultdict(Counter)
    per_read_tracks = defaultdict(set)
    if not os.path.exists(iou_patch_json):
        print(f"[WARN] IoU patch JSON not found: {iou_patch_json}")
        return per_read, per_read_class, per_read_tracks
    with open(iou_patch_json, 'r') as f:
        tracks = json.load(f)
    for tr in tracks:
        tid = tr.get('track_id')
        for c in tr.get('contours', []):
            sl = c.get('slice_index')
            r = slice_to_read.get(sl, None)
            if r is None:
                # fallback: unknown slide (won't be counted)
                continue
            cls = _norm(c.get('object',''))
            per_read[r] += 1
            per_read_class[r][cls] += 1
            per_read_tracks[r].add(tid)
    return per_read, per_read_class, per_read_tracks

def _count_from_iou_wsi(iou_wsi_json):
    """Counts per read_index and per class from IoU_Object_WSI.json (WSI space)."""
    per_read = Counter()
    per_read_class = defaultdict(Counter)
    per_read_tracks = defaultdict(set)
    if not os.path.exists(iou_wsi_json):
        print(f"[WARN] IoU WSI JSON not found: {iou_wsi_json}")
        return per_read, per_read_class, per_read_tracks
    with open(iou_wsi_json, 'r') as f:
        tracks = json.load(f)
    for tr in tracks:
        tid = tr.get('track_id')
        for c in tr.get('contours', []):
            r = c.get('read_index')
            if not r:
                continue
            r = _norm(r)
            cls = _norm(c.get('object',''))
            coords = c.get('coordinates') or []
            if not coords:
                # you might choose to skip 'empty' shapes; here we still count them
                pass
            per_read[r] += 1
            per_read_class[r][cls] += 1
            per_read_tracks[r].add(tid)
    return per_read, per_read_class, per_read_tracks

def _count_from_geojson_dir(geojson_dir):
    """Counts features per read_index (in filenames) and per class from exported GeoJSONs."""
    per_read = Counter()
    per_read_class = defaultdict(Counter)
    per_read_tracks = defaultdict(set)
    if not os.path.isdir(geojson_dir):
        print(f"[WARN] GeoJSON dir not found: {geojson_dir}")
        return per_read, per_read_class, per_read_tracks
    for fname in os.listdir(geojson_dir):
        if not fname.endswith('_tracks.geojson'):
            continue
        read_index = _norm(fname.replace('_tracks.geojson',''))
        path = os.path.join(geojson_dir, fname)
        try:
            with open(path, 'r') as f:
                fc = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue
        feats = fc.get('features', [])
        per_read[read_index] += len(feats)
        for feat in feats:
            props = feat.get('properties', {}) or {}
            cls = _norm(((props.get('classification') or {}).get('name')) or '')
            per_read_class[read_index][cls] += 1
            tid = props.get('measurements',{}).get('track_id')
            if tid is not None:
                per_read_tracks[read_index].add(tid)
    return per_read, per_read_class, per_read_tracks

def _fmt_diff(a, b):
    return "" if a == b else f" (!) {a} vs {b}"

def check_per_slide_counts(iou_patch_json, iou_wsi_json, csv2_path, geojson_dir=None, show_per_class=True, show_track_sets=False):
    """
    Compare per-slide counts across:
      - IoU_Object.json (patch)  -> via slice_index -> read_index from csv2
      - IoU_Object_WSI.json (WSI)
      - GeoJSON dir (optional)
    """
    slice_to_read = _load_slice_to_read_from_csv2(csv2_path)

    p_cnt, p_cls, p_tracks = _count_from_iou_patch(iou_patch_json, slice_to_read)
    w_cnt, w_cls, w_tracks = _count_from_iou_wsi(iou_wsi_json)
    g_cnt = g_cls = g_tracks = None
    if geojson_dir:
        g_cnt, g_cls, g_tracks = _count_from_geojson_dir(geojson_dir)

    # union of all read_index seen
    all_reads = set(p_cnt.keys()) | set(w_cnt.keys())
    if g_cnt is not None:
        all_reads |= set(g_cnt.keys())
    all_reads = sorted(all_reads, key=lambda x: (len(x), x))

    print("\n=== Per-slide (read_index) total item counts ===")
    print(f"{'read_index':>10}  {'patch_json':>10}  {'wsi_json':>10}" + (f"  {'geojson':>10}" if g_cnt is not None else ""))
    for r in all_reads:
        pc = p_cnt.get(r, 0)
        wc = w_cnt.get(r, 0)
        line = f"{r:>10}  {pc:>10}  {wc:>10}{_fmt_diff(pc, wc)}"
        if g_cnt is not None:
            gc = g_cnt.get(r, 0)
            mismatch = "" if (gc == wc == pc) else f" (!) {gc} vs patch/wsi"
            line += f"  {gc:>10}{mismatch}"
        print(line)

    if show_per_class:
        print("\n=== Per-slide per-class counts (patch_json vs wsi_json" + (", geojson" if g_cnt else "") + ") ===")
        for r in all_reads:
            classes = set(p_cls[r].keys()) | set(w_cls[r].keys())
            if g_cnt is not None:
                classes |= set(g_cls[r].keys())
            if not classes:
                continue
            print(f"\n[read_index: {r}]")
            print(f"{'class':>16}  {'patch':>8}  {'wsi':>8}" + (f"  {'geojson':>8}" if g_cnt else ""))
            for cls in sorted(classes):
                pv = p_cls[r].get(cls, 0)
                wv = w_cls[r].get(cls, 0)
                row = f"{cls:>16}  {pv:>8}  {wv:>8}{_fmt_diff(pv, wv)}"
                if g_cnt:
                    gv = g_cls[r].get(cls, 0)
                    row += f"  {gv:>8}{_fmt_diff(pv, gv) or _fmt_diff(wv, gv)}"
                print(row)

    if show_track_sets:
        print("\n=== Per-slide unique track_id counts (patch_json vs wsi_json" + (", geojson" if g_cnt else "") + ") ===")
        for r in all_reads:
            pt = len(p_tracks[r])
            wt = len(w_tracks[r])
            line = f"{r:>10}  tracks: patch={pt}, wsi={wt}{_fmt_diff(pt, wt)}"
            if g_cnt:
                gt = len(g_tracks[r])
                line += f", geojson={gt}{_fmt_diff(wt, gt) or _fmt_diff(pt, gt)}"
            print(line)

    print("\n[Done] If you see (!) on a row, counts differ across sources for that slide.")

# ---------------------------
# Command line
# ---------------------------
if __name__ == "__main__":
    args, cfg = parse_args(__doc__, lambda p: p.add_argument(
        "--tracks-suffix", "--suffix", dest="suffix", default="",
        help="suffix used by track_objects.py (e.g. '_longglom' after --min-track-len 10)"))
    for case in list_cases(cfg):
        case_contour_dir = os.path.join(cfg['paths']['contour_root'], os.path.basename(case))
        iou_dir = os.path.join(case_contour_dir, "Object_IoU_patch_connectglom")

        iou_patch_json = os.path.join(iou_dir, "IoU_Object%s.json" % args.suffix)
        iou_wsi_json   = os.path.join(iou_dir, "IoU_Object_WSI%s.json" % args.suffix)
        csv2_path      = os.path.join(case_contour_dir,
                                      "registration_3D_patch_ANTs", "annotation_QA_generegions",
                                      "patch_contours_before_after.csv")
        geojson_dir    = os.path.join(iou_dir, "WSI_tracking_geojson%s" % args.suffix)  # optional

        print("\n######## case %s" % os.path.basename(case))
        check_per_slide_counts(iou_patch_json, iou_wsi_json, csv2_path, geojson_dir,
                               show_per_class=True, show_track_sets=True)
