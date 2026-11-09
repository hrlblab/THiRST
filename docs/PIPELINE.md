# THiRST pipeline: data layout and step-by-step reference

Every script reads one YAML config (`--config`, see [`configs/example.yaml`](../configs/example.yaml))
describing **one tissue block**: a stack of serial sections whose WSIs are named `<case>-<section>.svs`.
`<case>` is the block identifier (the first two `-`-separated fields of the WSI name, e.g. `CASE-01`)
and `<section>` the section number. The association and object-level scripts assume that Visium V2 and
Visium HD sections alternate, starting with V2 (the 1st, 3rd, 5th, ... sections of the ordered stack are
V2, the 2nd, 4th, ... HD), as in the paper; they check this against `st_sections` and stop otherwise.

A step that cannot produce complete results stops with an error rather than writing partial output
(failed ANTs pairs, missing transforms, missing per-section files, misaligned tables); fix the cause
and re-run that step before continuing.

## Conventions

* **Two pixel frames.** *Full resolution* = level 0 of the WSI (Space Ranger `pxl_*_in_fullres`,
  QuPath/Omni-Seg GeoJSON). *4x frame* = the section image used for registration: pyramid level 2
  (assumed 4× downsampled) of the tissue region, padded by 1024 px on every side and saved as
  `image_<section>_<x0>_<y0>.png`, where `(x0, y0)` is the full-resolution position of its top-left pixel.
  Hence `col_full = x0 + 4 * col_4x` and `row_full = y0 + 4 * row_4x`.
* **ST unit tables** (`*_center.csv`): `barcode, abs_x, abs_y, now_x, now_y` with `abs_x`/`now_x` = **row**
  and `abs_y`/`now_y` = **column** (full resolution / 4x frame). Polygon tables (`*_spot.csv`) hold
  `now_x_k` (row) and `now_y_k` (column) for every vertex: 100 vertices for V2 spots, 4 for HD bins.
* **GeoJSON** coordinates are `[x = column, y = row]`.
* **Reference section.** The section at 0-based position `floor(N/2)` of the sorted stack is the
  reference; every other section is mapped onto it by chaining the pairwise affines (ANTs convention:
  each `output0GenericAffine.mat` maps fixed-image coordinates to moving-image coordinates; the pair
  `<i+1>_to_<i>` has section `i+1` as moving and `i` as fixed image).
* **Registration stages.** Outputs under `registration_3D_ANTs/` use the global affine chain;
  outputs under `registration_3D_patch_ANTs/` are additionally refined with the nearest local patch.

## Folder layout

Inputs (paths from the config):

```
<wsi_dir>/<case>-<section>.svs
<region_annotation_dir>/*<case>-<section>.svs.geojson      # polygon of class `region_class`
<object_annotation_dir>/<case>-<section>.svs.geojson       # Omni-Seg + QuPath QA, classes in `object_classes`
                                                           # (one Polygon/MultiPolygon per object; holes ignored)
Space Ranger outputs listed under `st_sections`
```

Created by the pipeline (one sub-folder per case in each of the three working roots):

```
<image_root>/<case>/
├── image_<section>_<x0>_<y0>.png                 # stage 0 (--save-png)
├── registration_xfeat/<i+1>_to_<i>/              # XFeat affine (sg_affine_init.npy) + QC images
├── registration_ANTs/<i+1>_to_<i>/               # sg_affine_init.mat, step1_prepare_data/, step2_run_ants_reg/
├── registration_3D_ANTs/image_*.png              # sections registered to the reference
└── 3D_Patch_crop/<abs_x>_<abs_y>/                # one local stack per sampled patch centre
    ├── image_<section>.png
    ├── registration_xfeat/ registration_ANTs/    # patch-level registration
    └── registration_3D_ANTs/image_<section>.png  # locally refined patch stack

<point_root>/<case>/
├── image_<section>_<x0>_<y0>_center.csv          # stage 0
├── <reference image>_center_sample.csv           # patch centres (sample_patch_centers.py)
├── registration_3D_ANTs/*_center.csv             # global transfer
└── registration_3D_patch_ANTs/*_center.csv       # patch-refined transfer

<contour_root>/<case>/
├── image_<section>_<x0>_<y0>_spot.csv            # stage 0
├── registration_3D_ANTs/*_spot.csv, annotation_QA/                       # global transfer
├── registration_3D_patch_ANTs/*_spot.csv, annotation_QA_generegions/     # patch-refined transfer
├── IoU_<patch|global|no>reg/                     # V2-V2 spot IoU
├── IoU_ST_to_HD_distance_<stage>reg[_merge]/     # V2-HD IoU
├── Object_IoU_patch_connectglom/                 # 3D objects (track_objects.py)
├── Object_IoU_patch/ | Object_IoU_global/        # 3D objects (track_objects_greedy.py)
└── Object_tracking_IoU_affine_*/, qupath_annotations_by_track_clusters/  # stage 5-6 outputs
```

## Steps

### Stage 0 — preprocessing (`scripts/0_preprocessing`)

| Script | Reads | Writes |
|---|---|---|
| `export_st_centers.py [--save-png]` | WSIs, region GeoJSONs, `st_sections` | `<point_root>/<case>/image_*_center.csv`; with `--save-png` the section images in `<image_root>/<case>/` |
| `export_st_contours.py` | same | `<contour_root>/<case>/image_*_spot.csv` (V2: circle r = 28 px ≈ 55 µm; HD: 8 µm square) |

Only in-tissue units inside the region polygon are kept; when `in_tissue_positions` is given, its
`in_tissue` flag is matched to `positions` by barcode. Both tables list the same units in the same order.

### Stage 1 — registration (`scripts/1_registration`)

Run the four registration scripts first with `--level global` (stacks `<image_root>/<case>/`) and, after
patch sampling and cropping, with `--level patch` (stacks `<image_root>/<case>/3D_Patch_crop/*/`).

| Script | Method | Writes (inside each stack) |
|---|---|---|
| `xfeat_affine.py` | XFeat keypoints (top-k 4096, 1000 px images) → USAC-MAGSAC + RANSAC → affine | `registration_xfeat/<i+1>_to_<i>/sg_affine_init.npy`, `registration_xfeat/xfeat_status.csv` |
| `affine_to_itk.py` | invert and convert to an ITK affine centred on the image | `registration_ANTs/<i+1>_to_<i>/sg_affine_init.mat` |
| `ants_affine.py [--overwrite \| --trust-existing]` | `antsRegistrationSyN.sh -t s` (MI, multi-resolution) initialised with the XFeat affine | `registration_ANTs/<i+1>_to_<i>/step2_run_ants_reg/output0GenericAffine.mat`, `ants_completed.json` |
| `warp_to_reference.py` | chain the affines to the reference section and resample | `registration_3D_ANTs/image_*.png` |
| `sample_patch_centers.py [--min-distance 512]` | greedy KD-tree sampling of reference-section ST centres | `<point_root>/<case>/<ref>_center_sample.csv` |
| `crop_patch_stacks.py` | 2048 × 2048 px (4x frame) windows around every centre, all sections | `3D_Patch_crop/<abs_x>_<abs_y>/image_<section>.png` |

`xfeat_status.csv` gives `ok`, `identity_fallback` (≤ 10 inlier matches: ANTs starts from the
unaligned pair; check the result) or `failed` (no initialisation; the script exits with status 1) for
every pair. `xfeat_affine.py` also writes `<image_root>/xfeat_provenance_<level>.json`: the source and
ref from `xfeat_hub`, the git commit when known (pinned commit or local clone), SHA-256 of the code
files that were run and of the weights, and the torch version; with `xfeat_weights_sha256` set, other
weights stop the run. `ants_affine.py` exits with status 1 and lists the pairs for which ANTs failed
or wrote no readable affine. A successful pair gets `ants_completed.json` (command, ANTs version,
SHA-256 of the affine); a failed one leaves nothing in `step2_run_ants_reg/` (its partial outputs go to
`step2_run_ants_reg.failed/`). On a restart only pairs with a matching completion record are skipped;
`--trust-existing` also keeps valid affines without one (e.g. from the original scripts). Only the affine part of
the ANTs result is used downstream; the SyN warp fields (`output1Warp.nii.gz`) are not applied by any
script.

### Stage 2 — coordinate transfer (`scripts/2_coordinate_transfer`)

| Script | Writes |
|---|---|
| `global_spot_centers.py`, `global_spot_contours.py` | `registration_3D_ANTs/*_center.csv` / `*_spot.csv` |
| `global_objects.py` | `registration_3D_ANTs/annotation_QA/<case>-<section>.svs.geojson` + `contours_before_after.csv` |
| `patch_spot_centers.py`, `patch_spot_contours.py` | `registration_3D_patch_ANTs/*_center.csv` / `*_spot.csv` (every unit follows the chain of its nearest patch stack) |
| `patch_objects.py` | `registration_3D_patch_ANTs/annotation_QA_generegions/*.geojson` + `patch_contours_before_after.csv` |

Every section is mapped through all pairwise transforms between it and the reference section. Before
writing anything, each script checks that every adjacent pair of the stack (and, for the `patch_*`
scripts, of every patch stack) has a readable affine, and stops with the list of missing pairs
otherwise. The `*_spot_contours.py` scripts take V2 spots or HD bins from the polygon table (100
vertices or 4) and stop if that disagrees with `st_sections`. `patch_objects.py` leaves out (and counts,
per section) contours that extend beyond the 2048 px window of their nearest patch stack.

The object files of stage 2 are standard GeoJSON: one `Polygon` feature per object, with its class and
`contour_id` (the id used in the before/after CSVs) as properties, so they open in QuPath and any GIS
library. Files written by earlier versions (one feature per class holding all contours) are still read.

### Stage 3 — ST unit association (`scripts/3_st_association`)

All three accept `--registration patch|global|none` to compute the correspondences after patch
refinement, after the global chain only, or on the unregistered sections.

| Script | Writes |
|---|---|
| `v2_v2_iou.py` | `IoU_<stage>reg/row_<i>_col_<j>.json` — for every pair of V2 sections, `{barcode_i: [{barcode_j, iou}]}` for all non-zero overlaps |
| `v2_hd_iou.py` | `IoU_ST_to_HD_distance_<stage>reg/row_<i>_col_<j>_<batch>.json` — V2 spot × HD bin IoUs, batches skipped when all centroids are > 30 px apart |
| `merge_v2_hd_iou.py` | `IoU_ST_to_HD_distance_<stage>reg_merge/row_<i>_col_<j>.json` |

Units whose `now_x` is 0 (no coordinates) are left out; the IoUs are stored under the barcodes of the
overlapping units. The `*_center.csv` and `*_spot.csv` tables of a section must list the same barcodes
in the same order (checked). These sparse IoU tables are the input of the cross-section
reproducibility analysis (one-to-one matching, cluster concordance C(d) and expression correlation
R(d)) and of the HD→V2 aggregation, which are not part of this repository.

### Stage 4 — 3D objects (`scripts/4_object_tracking`)

| Script | Writes |
|---|---|
| `track_objects.py [--min-track-len N] [--iou-threshold 0.01]` | `Object_IoU_patch_connectglom/IoU_Object{suffix}.json`, `IoU_Object_WSI{suffix}.json`, `track_centers_by_id{suffix}.csv`, `WSI_tracking_geojson{suffix}/` |
| `track_objects_greedy.py [--registration patch\|global]` | `Object_IoU_patch/IoU_Object.json` or `Object_IoU_global/IoU_Object.json` |
| `check_track_counts.py [--tracks-suffix]` | consistency report (stdout) |
| `cross_section_area.py [--tracks-subdir] [--tracks-suffix] [--pixel-spacing]` | `cross_section_areas_per_contour{suffix}.csv`, `cross_section_areas_per_slice{suffix}.csv`, in the folder of the tracks read (default: greedy, `Object_IoU_patch/`) |

Both trackers link only objects of the same class. They read one GeoJSON per section,
`<case>-<section>.svs.geojson`, and stop if a section has
none (a section without objects needs an empty FeatureCollection). `slice_index` is the position of the
section in the ordered stack. Objects cannot be linked across a section without objects; the trackers
name such sections and objects after them start new tracks.

`IoU_Object.json` is a list of tracks `{"track_id", "contours": [{"slice_index", "object", "index", "coordinates"}]}`
in the 4x reference frame; `IoU_Object_WSI.json` gives the same contours in every WSI's own coordinates
(import `WSI_tracking_geojson/*.geojson` into QuPath to inspect them).

### Stage 5 — object-level ST (`scripts/5_object_analysis`)

Scripts that read tracks accept `--tracks-subdir` to choose `Object_IoU_patch_connectglom` (Hungarian
tracker) or `Object_IoU_patch` (greedy tracker). The defaults are the folders that the corresponding
scripts of the second original working copy read when they were last saved; they do not show which
tracker produced a published result (the first working copy read the other tracker in two of these
scripts, see [`PROVENANCE.md`](PROVENANCE.md#which-tracks-each-analysis-read)), so pass the folder
explicitly. Output folder names carry a tag `<tag>` that says which tracks were read, as in the original
scripts: `_connectglom` for `Object_IoU_patch_connectglom`, nothing for `Object_IoU_patch`, and
`_<folder>` for any other folder (the tables below show it in place).
These scripts (and those of stage 6 that read tracks) support exactly one entry in `object_classes`.
`--tracks-suffix _longglom` makes them (and `cross_section_area.py`, `check_track_counts.py`) read the
tracks of `track_objects.py --min-track-len N` (`IoU_Object_longglom.json`); their output folders then
carry the same suffix. A missing track file is an error. Slice indices are mapped to the `sample` names
of `st_sections` in stack order.

| Script | Extra inputs | Writes (under `<contour_root>/<case>/`) |
|---|---|---|
| `match_objects_to_spots.py` (default: Hungarian tracks) | — | `Object_tracking_IoU_affine_patch<tag>/<track>/Glo_slice_<k>.png`, `IoU_withspot.csv` |
| `export_track_barcodes.py` (default: Hungarian) | — | `Object_tracking_IoU_affine_patch_gene_barcode<tag>/glomeruli_barcode_per_slice.json` |
| `gene_maps_per_track.py` (default: greedy) | `analysis.hd_expression_mtx`, `hd_barcodes_csv`, `hd_genes_csv`, `marker_gene_list_csv` | `Object_tracking_IoU_affine_patch_gene_highselect_colorbar_yanglist<tag>/<track>/<gene>/` |
| `qupath_cells_by_cluster.py` (default: greedy) | `analysis.clustering_meta_csv`, `cell_clusters_per_track` | `qupath_annotations_by_track_clusters<tag>/track_<id>/*.geojson` |
| `qupath_all_cells_by_cluster.py [--out-dir]` | `analysis.clustering_meta_csv`, `cell_clusters_all` | `<sample>_allcells.geojson` per HD section |

Both QuPath scripts match HD barcodes in the short form `XXXXX_YYYYY-1`, whatever form the spot tables
and the clustering table use (`s_008um_XXXXX_YYYYY-1`, `<sample>-8-XXXXX_YYYYY-1`, ...).

### Stage 6 — figures and QC (`scripts/6_visualization`)

| Script | Writes |
|---|---|
| `checkerboard.py --mode global\|patch-before\|patch-after` | RGB/gray checkerboards of every section pair (registration QC) |
| `tracking_figure.py` (default: Hungarian tracks) | registered patch stacks that contain tracked objects (`Object_tracking_IoU_affine_patch<tag>_global_vis/`) |
| `render_3d_objects.py` (default: greedy tracks) | per-object tissue crops on white background, 3D renderings (`Object_tracking_IoU_affine_patch_tissueviual_white<tag>/`) |
| `qc_spot_sampling.py` | sampled patch centres and their spots drawn on the reference section |
