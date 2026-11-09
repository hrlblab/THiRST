"""Link segmented objects across sections into 3D objects (Hungarian IoU tracking) and export them.

THiRST stage 4 - 3D object reconstruction and cross-slice association (paper Sec. 4.2.5, Algorithm 3).

Contours of adjacent sections (patch-refined reference frame) are matched one-to-one with the Hungarian
algorithm on 1 - IoU; only contours of the same class are linked, pairs with IoU < ``--iou-threshold``
(0.01) are rejected and unmatched contours start new tracks. Every contour is also traced back to its source WSI using the before/after CSVs of
``global_objects.py`` and ``patch_objects.py``.

Outputs (``<contour_root>/<case>/Object_IoU_patch_connectglom/``):
  IoU_Object{suffix}.json        tracks in the reference frame (input of scripts 5_* and 6_*)
  IoU_Object_WSI{suffix}.json    the same tracks in each WSI's own coordinates
  track_centers_by_id{suffix}.csv
  WSI_tracking_geojson{suffix}/<section>_tracks{suffix}.geojson   (import into QuPath)

``--min-track-len 10`` keeps only objects seen on >= 10 sections ("long glomeruli", suffix _longglom).

Every section needs its GeoJSON ``<case>-<section>.svs.geojson`` (an empty FeatureCollection if it has
no object); a missing file stops the script. Objects are not linked across a section without objects:
such sections are named in a warning and the objects after them start new tracks.

Usage
-----
    python scripts/4_object_tracking/track_objects.py --config configs/example.yaml [--min-track-len 10]

Original script(s): Step11_Match_Object_tracking_results_QuPath_allglom_0915.py and Step11_Match_Object_tracking_results_QuPath_onlylongglom_0915.py (successors of the Step9_contourMatching_object_new_connectglom* scripts)
"""
import sys
import os
import glob
import re
import json
import csv

import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args, section_index, section_object_geojsons  # noqa: E402
from thirst.objects import read_object_contours  # noqa: E402


# -----------------------
# Settings (data roots and MASK_LABELS come from --config; the rest can be set on the command line)
# -----------------------
CASE_FOLDER     = None   # paths.image_root
CONTOUR_FOLDER  = None   # paths.contour_root
MASK_LABELS     = ['capsule']   # object_classes
IOU_THRESHOLD   = 0.01
ATOL_SIG        = 1e-5   # signature quantization (INTERMEDIATE & FINAL)
FORCE_NEAREST   = True   # force nearest-centroid fallback for WSI chaining
WRITE_GEOJSON   = True   # write per-slide GeoJSONs
MIN_TRACK_LEN   = 0      # keep tracks with >= MIN_TRACK_LEN contours (0 = keep all)
SUFFIX          = ""     # appended to every output file name

# -----------------------
# Utilities
# -----------------------
def atoi(text): return int(text) if text.isdigit() else text
def natural_keys(text): return [atoi(c) for c in re.split(r'(\d+)', text)]
def _norm(s): return str(s).strip().lower()

def compute_iou(poly1, poly2):
    # Fast bbox reject
    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds
    if (maxx1 < minx2) or (maxx2 < minx1) or (maxy1 < miny2) or (maxy2 < miny1):
        return 0.0
    inter = poly1.intersection(poly2)
    if inter.is_empty: return 0.0
    inter_area = inter.area
    union_area = poly1.area + poly2.area - inter_area
    return inter_area / union_area if union_area else 0.0

def _sig_points(pts_xy, atol=ATOL_SIG):
    n = len(pts_xy)
    if n == 0: return (0,)
    xs = [p[0] for p in pts_xy]; ys = [p[1] for p in pts_xy]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    # area & perimeter assuming closed ring
    per = 0.0; area2 = 0.0
    for i in range(n):
        x1,y1 = pts_xy[i]; x2,y2 = pts_xy[(i+1)%n]
        per += ((x2-x1)**2 + (y2-y1)**2)**0.5
        area2 += (x1*y2 - x2*y1)
    area = abs(area2)*0.5
    # 9 spread samples
    picks = [0, int(0.125*(n-1)), int(0.25*(n-1)), int(0.375*(n-1)),
             int(0.5*(n-1)), int(0.625*(n-1)), int(0.75*(n-1)),
             int(0.875*(n-1)), n-1]
    picks = [max(0, min(n-1, j)) for j in picks]
    def q(v): return int(round(v/atol)) if atol and atol>0 else v
    sig = [n, q(minx), q(miny), q(maxx), q(maxy), q(area), q(per)]
    for j in picks:
        x,y = pts_xy[j]; sig += [q(x), q(y)]
    return tuple(sig)

def _centroid(pts_xy):
    if not pts_xy: return (None, None)
    xs = [p[0] for p in pts_xy]; ys = [p[1] for p in pts_xy]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

# -----------------------
# Read per-slice GeoJSON (patch space)
# -----------------------
def load_annotations(mask_json_path, mask_labels):
    """
    Returns list of dicts per contour:
      {'object', 'index', 'polygon'}
    """
    if not os.path.exists(mask_json_path):
        print(f"Skipping {mask_json_path}, file not found")
        return None

    contours_by_class = read_object_contours(mask_json_path, mask_labels)

    slice_contours = []
    for obj in mask_labels:
        for idx, contour in enumerate(contours_by_class[obj]):
            pts = np.array(contour, dtype=np.float64)
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            slice_contours.append({'object': obj, 'index': idx, 'polygon': poly})
    return slice_contours

# -----------------------
# csv1 + csv2 -> WSI map (INTERMEDIATE exact + nearest-centroid fallback)
# -----------------------
def load_chained_wsi_map(csv1_path, csv2_path, atol=ATOL_SIG, force_nearest=FORCE_NEAREST):
    if not os.path.exists(csv2_path):
        print(f"[WARN] csv2 not found: {csv2_path}"); return {}, {}
    if not os.path.exists(csv1_path):
        print(f"[WARN] csv1 not found: {csv1_path}"); return {}, {}

    def read_rows(path):
        with open(path, 'r') as f:
            header = f.readline().strip().split(',')
            idx = {name:i for i,name in enumerate(header)}
            rows = []
            for line in f:
                s = line.strip()
                if not s: continue
                rows.append((idx, s.split(',')))
            return idx, rows

    idx1, rows1 = read_rows(csv1_path)
    idx2, rows2 = read_rows(csv2_path)

    def bucket(rows, idx):
        b = {}
        for _i, p in rows:
            key = (_norm(p[idx['read_index']]), _norm(p[idx['class']]), _norm(p[idx['contour_id']]))
            pt_idx = int(p[idx['point_idx']])
            x_orig = float(p[idx['x_orig']]); y_orig = float(p[idx['y_orig']])
            x_new  = float(p[idx['x_new']]) if 'x_new' in idx else None
            y_new  = float(p[idx['y_new']]) if 'y_new' in idx else None
            b.setdefault(key, []).append((pt_idx, x_orig, y_orig, x_new, y_new))
        for k in b: b[k].sort(key=lambda t: t[0])
        return b

    b1 = bucket(rows1, idx1)  # (pt_idx, x_wsi, y_wsi, x_mid, y_mid)
    b2 = bucket(rows2, idx2)  # (pt_idx, x_mid, y_mid, x_final, y_final)

    # index csv1 by INTERMEDIATE signature and by (read,class) group
    idx1_sig = {}
    idx1_group = {}
    for (read_idx, cls, cid1), pts in b1.items():
        mid_pts = [(x_mid, y_mid) for _, _, _, x_mid, y_mid in pts]
        wsi_pts = [(x_wsi, y_wsi) for _, x_wsi, y_wsi, _, _ in pts]
        sig = _sig_points(mid_pts, atol)
        idx1_sig.setdefault((read_idx, cls, sig), []).append((cid1, wsi_pts))
        cmx, cmy = _centroid(mid_pts)
        idx1_group.setdefault((read_idx, cls), []).append((cid1, wsi_pts, (cmx, cmy)))

    wsi_map = {}
    meta = {}
    exact = fallback = missing = ambiguous = 0
    for (read_idx, cls, cid2), pts in b2.items():
        mid_pts2 = [(x_mid, y_mid) for _, x_mid, y_mid, _, _ in pts]
        sig2 = _sig_points(mid_pts2, atol)
        cmx2, cmy2 = _centroid(mid_pts2)

        cands = idx1_sig.get((read_idx, cls, sig2), [])
        if len(cands) == 1:
            _cid1, wsi_pts = cands[0]
            wsi_map[(read_idx, cls, cid2)] = wsi_pts
            meta[(read_idx, cls, cid2)] = {'match': 'exact', 'mid_fallback_dist': 0.0}
            exact += 1
            continue
        elif len(cands) > 1:
            ambiguous += 1

        if force_nearest:
            group = idx1_group.get((read_idx, cls), [])
            if group and (cmx2 is not None):
                best = None; best_d2 = None
                for cid1, wsi_pts, (cmx1, cmy1) in group:
                    if cmx1 is None: continue
                    dx = cmx1 - cmx2; dy = cmy1 - cmy2
                    d2 = dx*dx + dy*dy
                    if (best_d2 is None) or (d2 < best_d2):
                        best_d2 = d2; best = (cid1, wsi_pts)
                if best is not None:
                    _cid1, wsi_pts = best
                    wsi_map[(read_idx, cls, cid2)] = wsi_pts
                    meta[(read_idx, cls, cid2)] = {'match': 'fallback', 'mid_fallback_dist': float(best_d2**0.5)}
                    fallback += 1
                    continue

        meta[(read_idx, cls, cid2)] = {'match': 'missing', 'mid_fallback_dist': ''}
        missing += 1

    print(f"[INFO] WSI map: total={len(b2)} | exact={exact} | fallback_nearest={fallback} | ambiguous={ambiguous} | missing={missing} (sig atol={atol})")
    return wsi_map, meta

# -----------------------
# Build csv2 FINAL index -> resolve proper contour_id by geometry
# -----------------------
def build_csv2_final_index(csv2_path, atol_final=ATOL_SIG):
    """
    Returns:
      sig2final[(read_index_norm, class_norm, sig_final)] = (contour_id_norm, slice_idx, image_name)
      final_groups[(read_index_norm, class_norm)] = list of (contour_id_norm, slice_idx, image_name, (cmx, cmy))
      slice_to_read[slice_idx:int] = read_index_norm
    """
    sig2final = {}
    final_groups = {}
    slice_to_read = {}

    if not os.path.exists(csv2_path):
        print(f"[WARN] csv2 not found: {csv2_path}")
        return sig2final, final_groups, slice_to_read

    with open(csv2_path, 'r') as f:
        header = f.readline().strip().split(',')
        idx = {name:i for i,name in enumerate(header)}
        need = ['read_index','class','contour_id','slice_idx','image_name','point_idx','x_new','y_new']
        for k in need:
            if k not in idx:
                raise RuntimeError(f"csv2 missing column '{k}'")

        # bucket all FINAL points per contour
        buckets = {}
        for line in f:
            s = line.strip()
            if not s: continue
            p = s.split(',')
            read_idx = _norm(p[idx['read_index']])
            cls = _norm(p[idx['class']])
            cid = _norm(p[idx['contour_id']])
            slice_idx = p[idx['slice_idx']]
            image_name = p[idx['image_name']]
            pt_idx = int(p[idx['point_idx']])
            xf = float(p[idx['x_new']]); yf = float(p[idx['y_new']])
            key = (read_idx, cls, cid)
            buckets.setdefault(key, {'pts':[], 'slice_idx':slice_idx, 'image_name':image_name})
            buckets[key]['pts'].append((pt_idx, xf, yf))

        # finalize signatures and centroids
        for key, data in buckets.items():
            (read_idx, cls, cid) = key
            pts = [xy for _, *xy in sorted(data['pts'], key=lambda t:t[0])]
            sig = _sig_points(pts, atol_final)
            cmx, cmy = _centroid(pts)
            sig2final[(read_idx, cls, sig)] = (cid, data['slice_idx'], data['image_name'])
            final_groups.setdefault((read_idx, cls), []).append((cid, data['slice_idx'], data['image_name'], (cmx, cmy)))
            try:
                slice_to_read[int(data['slice_idx'])] = read_idx
            except:
                pass

    return sig2final, final_groups, slice_to_read

# -----------------------
# Save tracks (patch space)
# -----------------------
def save_tracks_to_json(tracks, output_file):
    out = []
    for tid, track in tracks.items():
        entry = {'track_id': tid, 'contours': []}
        for c in track:
            coords = list(c['polygon'].exterior.coords) if c['polygon'].exterior else []
            entry['contours'].append({
                'slice_index': c['slice_index'],
                'object': c['object'],
                'index': c['index'],
                'coordinates': coords
            })
        out.append(entry)
    with open(output_file, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"[OK] Tracks (patch coords) saved to {output_file}")

# -----------------------
# Centers CSV (patch + WSI + QA)
# -----------------------
def write_track_centers_csv(now_case, tracks, images, wsi_map, wsi_meta,
                            sig2final, final_groups, slice_to_read,
                            out_csv_path, atol_final=ATOL_SIG):
    rows = []
    for tid, tr in tracks.items():
        for c in tr:
            poly = c['polygon']
            cp = poly.centroid
            cx_patch, cy_patch = float(cp.x), float(cp.y)

            pts = list(poly.exterior.coords) if poly.exterior else []
            pts = [(float(x), float(y)) for (x,y) in pts]

            slice_idx = c['slice_index']
            image_name = os.path.basename(images[slice_idx]) if 0 <= slice_idx < len(images) else ''
            read_idx_norm = slice_to_read.get(slice_idx, None)
            cls_norm = _norm(c['object'])

            # try FINAL exact
            cid_norm = None; img_for_row = image_name
            if read_idx_norm is not None:
                sigF = _sig_points(pts, atol_final)
                hit = sig2final.get((read_idx_norm, cls_norm, sigF))
                if hit:
                    cid_norm, _slice_csv2, img_for_row = hit

            # FINAL fallback
            final_match_kind = 'exact'
            final_fallback_dist = ''
            if cid_norm is None:
                final_match_kind = 'fallback'
                if read_idx_norm is None:
                    m = re.search(r'image_(\d+)', image_name)
                    read_idx_norm = _norm(m.group(1)) if m else _norm(str(slice_idx))
                group = final_groups.get((read_idx_norm, cls_norm), [])
                if group:
                    cmx, cmy = _centroid(pts)
                    best = None; best_d2 = None
                    for cid_g, _sl, _im, (gx, gy) in group:
                        if gx is None: continue
                        dx = gx - cmx; dy = gy - cmy
                        d2 = dx*dx + dy*dy
                        if (best_d2 is None) or (d2 < best_d2):
                            best_d2 = d2; best = (cid_g, _im)
                    if best is not None:
                        cid_norm, img_for_row = best
                        final_fallback_dist = float(best_d2**0.5)

            # WSI map
            cx_wsi = cy_wsi = ''
            wsi_kind = 'missing'
            wsi_fallback_d = ''
            if cid_norm is not None and read_idx_norm is not None:
                key = (read_idx_norm, cls_norm, cid_norm)
                wsi_pts = wsi_map.get(key)
                meta = wsi_meta.get(key, {'match':'missing','mid_fallback_dist':''})
                if wsi_pts:
                    xs = [p[0] for p in wsi_pts]; ys = [p[1] for p in wsi_pts]
                    cx_wsi = sum(xs)/len(xs); cy_wsi = sum(ys)/len(ys)
                wsi_kind = meta['match']; wsi_fallback_d = meta['mid_fallback_dist']

            rows.append([
                now_case, tid, slice_idx, image_name,
                read_idx_norm if read_idx_norm is not None else '',
                c['object'], c['index'],
                cid_norm if cid_norm is not None else '',
                cx_patch, cy_patch, cx_wsi, cy_wsi,
                final_match_kind, final_fallback_dist,
                wsi_kind, wsi_fallback_d
            ])

    rows.sort(key=lambda r: (int(r[1]), int(r[2]), _norm(r[5]), int(r[6]) if str(r[6]).isdigit() else 10**9))
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'case','track_id','slice_idx','image_name',
            'read_index(norm)','class','geojson_index','contour_id(csv2)',
            'cx_patch','cy_patch','cx_wsi','cy_wsi',
            'final_match','final_fallback_dist',
            'wsi_match','mid_fallback_dist'
        ])
    with open(out_csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"[OK] Track centers CSV written: {out_csv_path} | rows={len(rows)}")

# -----------------------
# Save WSI JSON (includes read_index so we can split by slide)
# -----------------------
def save_tracks_to_json_wsi_resolved(tracks, images,
                                     wsi_map,
                                     sig2final, final_groups, slice_to_read,
                                     out_file, atol_final=ATOL_SIG):
    def _local_sig(pts_xy): return _sig_points(pts_xy, atol_final)

    out = []
    missing, total = 0, 0

    for tid, track in tracks.items():
        entry = {'track_id': tid, 'contours': []}
        for c in track:
            total += 1
            poly = c['polygon']
            pts = list(poly.exterior.coords) if poly.exterior else []
            pts = [(float(x), float(y)) for (x,y) in pts]
            sigF = _local_sig(pts)

            slice_idx = c['slice_index']
            image_name = os.path.basename(images[slice_idx]) if 0 <= slice_idx < len(images) else ''
            read_idx_norm = slice_to_read.get(slice_idx, None)
            cls_norm = _norm(c['object'])

            # FINAL exact -> contour_id
            cid_norm = None
            if read_idx_norm is not None:
                hit = sig2final.get((read_idx_norm, cls_norm, sigF))
                if hit:
                    cid_norm, _slice_csv2, _ = hit

            # FINAL fallback: nearest FINAL centroid in same (read, class)
            if cid_norm is None:
                if read_idx_norm is None:
                    m = re.search(r'image_(\d+)', image_name)
                    read_idx_norm = _norm(m.group(1)) if m else _norm(str(slice_idx))
                group = final_groups.get((read_idx_norm, cls_norm), [])
                if group:
                    cmx, cmy = _centroid(pts)
                    best = None; best_d2 = None
                    for cid_g, _sl, _im, (gx, gy) in group:
                        if gx is None: continue
                        dx = gx - cmx; dy = gy - cmy
                        d2 = dx*dx + dy*dy
                        if (best_d2 is None) or (d2 < best_d2):
                            best_d2 = d2; best = cid_g
                    if best is not None:
                        cid_norm = best

            # fetch WSI coords
            coords = []
            if cid_norm is not None and read_idx_norm is not None:
                key = (read_idx_norm, cls_norm, cid_norm)
                coords = wsi_map.get(key, [])
                if coords and coords[0] != coords[-1]:
                    coords = coords + [coords[0]]
            if not coords:
                missing += 1

            entry['contours'].append({
                'slice_index': c['slice_index'],
                'read_index': read_idx_norm if read_idx_norm is not None else '',
                'object': c['object'],
                'index': c['index'],                         # GeoJSON index (reference)
                'contour_id': cid_norm if cid_norm is not None else '',
                'coordinates': coords
            })
        out.append(entry)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"[OK] Tracks (WSI coords) saved to {out_file} | mapped {total-missing}/{total}, missing {missing}")

# -----------------------
# IoU Hungarian tracking (unchanged)
# -----------------------
def match_tracks_hungarian(all_tracks, previous_tracks, current_contours, next_track_id, iou_threshold=IOU_THRESHOLD):
    prev = [tr[-1] for tr in previous_tracks.values()]
    track_ids = list(previous_tracks.keys())
    if not prev or not current_contours:
        for c in current_contours:
            c['track_id'] = next_track_id
            all_tracks[next_track_id] = [c]
            next_track_id += 1
        return all_tracks, next_track_id

    iou = np.zeros((len(prev), len(current_contours)), dtype=np.float32)
    for i, p in enumerate(prev):
        for j, q in enumerate(current_contours):
            if _norm(p['object']) != _norm(q['object']):
                continue            # objects of different classes are never linked (IoU stays 0)
            iou[i, j] = compute_iou(p['polygon'], q['polygon'])
    cost = 1.0 - iou
    rows, cols = linear_sum_assignment(cost)

    assigned_curr = set()
    for i, j in zip(rows, cols):
        if iou[i, j] >= iou_threshold:
            tid = track_ids[i]
            all_tracks[tid].append(current_contours[j])
            current_contours[j]['track_id'] = tid
            assigned_curr.add(j)

    for j, c in enumerate(current_contours):
        if j not in assigned_curr:
            c['track_id'] = next_track_id
            all_tracks[next_track_id] = [c]
            next_track_id += 1
    return all_tracks, next_track_id


def track_sections(slices, iou_threshold=IOU_THRESHOLD):
    """Track contours through ``slices`` (one list of contours per section, in stack order).

    ``slices[i]`` may be empty (no object on that section). Every section is visited, so an empty
    section never hides the sections after it; a track cannot continue across an empty section, so
    objects after it start new tracks. Returns ``(tracks, next_track_id)``.
    """
    tracks = {}
    next_track_id = 0
    for c in slices[0] if slices else []:
        tracks[next_track_id] = [c]
        next_track_id += 1

    for slice_index in range(1, len(slices)):
        prev_tracks = {tid: tr for tid, tr in tracks.items() if tr[-1]['slice_index'] == slice_index - 1}
        current = slices[slice_index]
        tracks, next_track_id = match_tracks_hungarian(tracks, prev_tracks, current, next_track_id, iou_threshold)
    return tracks, next_track_id

# -----------------------
# Track length filter
# -----------------------
def filter_tracks_by_length(tracks, min_len=MIN_TRACK_LEN):
    """Return a dict with only tracks whose number of contours >= min_len."""
    kept = {tid: tr for tid, tr in tracks.items() if len(tr) >= min_len}
    dropped = {tid: tr for tid, tr in tracks.items() if len(tr) < min_len}
    print(f"[FILTER] kept {len(kept)} tracks (>= {min_len}), dropped {len(dropped)} (< {min_len})")
    return kept

# -----------------------
# Exporter: split IoU_Object_WSI.json into one GeoJSON per read_index
# -----------------------
def ensure_closed_ring(coords):
    if not coords: return coords
    return coords if coords[0] == coords[-1] else coords + [coords[0]]

def wsi_tracks_to_geojson(iou_wsi_json_path, output_dir):
    """
    Read IoU_Object_WSI.json and write one GeoJSON per WSI (read_index).
    """
    if not os.path.exists(iou_wsi_json_path):
        raise FileNotFoundError(f"WSI tracks JSON not found: {iou_wsi_json_path}")

    with open(iou_wsi_json_path, 'r') as f:
        tracks_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    per_wsi = {}
    for track in tracks_data:
        track_id = int(track.get('track_id'))
        for c in track.get('contours', []):
            read_index = c.get('read_index')
            if not read_index:
                continue
            read_index = str(read_index)

            coords = c.get('coordinates', [])
            if not coords:
                continue
            ring = ensure_closed_ring(coords)
            obj = c.get('object', 'object')

            feat = {
                "type": "Feature",
                "properties": {
                    "name": str(track_id),
                    "classification": {"name": obj},
                    "measurements": {"track_id": track_id},
                    "slice_index": c.get("slice_index"),
                    "contour_index": c.get("index"),
                    "contour_id": c.get("contour_id", "")
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]}
            }
            per_wsi.setdefault(read_index, []).append(feat)

    out_files = []
    for read_index, features in per_wsi.items():
        fc = {"type": "FeatureCollection", "features": features}
        out_path = os.path.join(output_dir, f"{read_index}_tracks{SUFFIX}.geojson")
        with open(out_path, "w") as f:
            json.dump(fc, f, indent=2)
        out_files.append(out_path)
        print(f"[OK] Wrote {len(features)} features → {out_path}")

    if not out_files:
        print("[WARN] No features written (check IoU_Object_WSI.json).")
    return out_files

# -----------------------
# Main
# -----------------------
def _add_arguments(parser):
    parser.add_argument('--iou-threshold', type=float, default=IOU_THRESHOLD,
                        help='minimum IoU to link objects on adjacent sections (default: %(default)s)')
    parser.add_argument('--min-track-len', type=int, default=MIN_TRACK_LEN,
                        help='keep only 3D objects traced through at least this many sections '
                             '(0 = keep all; the "long glomeruli" analysis used 10)')
    parser.add_argument('--suffix', default=None,
                        help="suffix for all output files (default: '_longglom' when --min-track-len > 1, else '')")


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, _add_arguments)
    CASE_FOLDER = cfg['paths']['image_root']
    CONTOUR_FOLDER = cfg['paths']['contour_root']
    MASK_LABELS = list(cfg['object_classes'])
    IOU_THRESHOLD = args.iou_threshold
    MIN_TRACK_LEN = args.min_track_len
    SUFFIX = args.suffix if args.suffix is not None else ('_longglom' if MIN_TRACK_LEN > 1 else '')

    cases = list_cases(cfg)
    for case in cases:
        now_case = os.path.basename(case)

        # Inputs
        images = glob.glob(os.path.join(case, 'image_*.png')); images.sort(key=natural_keys)
        reconstruction_object_dir = os.path.join(
            case.replace(CASE_FOLDER, CONTOUR_FOLDER),
            'registration_3D_patch_ANTs', 'annotation_QA_generegions'
        )
        jsons = section_object_geojsons(images, reconstruction_object_dir, now_case)  # one per section, in stack order

        # Outputs dir
        IoU_dir = os.path.join(case.replace(CASE_FOLDER, CONTOUR_FOLDER), 'Object_IoU_patch_connectglom')
        os.makedirs(IoU_dir, exist_ok=True)

        # CSV paths for chaining
        csv2_path = os.path.join(
            case.replace(CASE_FOLDER, CONTOUR_FOLDER),
            'registration_3D_patch_ANTs', 'annotation_QA_generegions',
            'patch_contours_before_after.csv'
        )
        reconstruction_dir = os.path.join(
            case.replace(CASE_FOLDER, CONTOUR_FOLDER),
            'registration_3D_ANTs', 'annotation_QA'
        )
        csv1_path = os.path.join(reconstruction_dir, 'contours_before_after.csv')

        # Build WSI map (INTERMEDIATE exact + nearest-centroid fallback).
        wsi_map, wsi_meta = load_chained_wsi_map(csv1_path, csv2_path, atol=ATOL_SIG, force_nearest=FORCE_NEAREST)

        # Build csv2 FINAL index to resolve (read_index, class, contour_id) from patch polygons
        sig2final, final_groups, slice_to_read = build_csv2_final_index(csv2_path, atol_final=ATOL_SIG)

        # Build slice contours for tracking (patch space); slice_index = position of the section in
        # `images`, the convention of the before/after CSVs used to trace contours back to the WSIs.
        all_slices = []
        for idx, json_path in enumerate(jsons):
            sc = load_annotations(json_path, MASK_LABELS)
            for c in sc: c['slice_index'] = idx
            all_slices.append(sc)

        empty = [section_index(images[i]) for i, sc in enumerate(all_slices) if not sc]
        if len(empty) == len(all_slices):
            print(f"[WARN] No {MASK_LABELS} objects on any section of case {now_case}")
            continue
        if empty:
            print(f"[WARN] {now_case}: no {MASK_LABELS} objects on section(s) {empty}; "
                  f"objects cannot be linked across these sections (tracks restart after them)")

        # IoU tracking over every section (an empty section does not end the loop)
        tracks, next_track_id = track_sections(all_slices, IOU_THRESHOLD)

        # Optionally keep only long tracks (objects seen on >= MIN_TRACK_LEN sections)
        if MIN_TRACK_LEN > 1:
            tracks = filter_tracks_by_length(tracks, MIN_TRACK_LEN)

        # Save patch-space tracks (suffixed)
        out_patch = os.path.join(IoU_dir, f'IoU_Object{SUFFIX}.json')
        save_tracks_to_json(tracks, out_patch)

        # Save centers CSV (suffixed)
        centers_csv = os.path.join(IoU_dir, f'track_centers_by_id{SUFFIX}.csv')
        write_track_centers_csv(now_case, tracks, images,
                                wsi_map, wsi_meta,
                                sig2final, final_groups, slice_to_read,
                                centers_csv, atol_final=ATOL_SIG)

        # Save WSI-space tracks (suffixed; includes read_index per contour)
        out_wsi = os.path.join(IoU_dir, f'IoU_Object_WSI{SUFFIX}.json')
        save_tracks_to_json_wsi_resolved(tracks, images,
                                         wsi_map,
                                         sig2final, final_groups, slice_to_read,
                                         out_wsi, atol_final=ATOL_SIG)

        # Optional: export one GeoJSON per slide (suffixed folder + files)
        if WRITE_GEOJSON:
            out_dir = os.path.join(IoU_dir, f'WSI_tracking_geojson{SUFFIX}')
            try:
                wsi_tracks_to_geojson(out_wsi, out_dir)
            except FileNotFoundError as e:
                print(f"[SKIP] {e}")
