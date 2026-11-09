# Code provenance

This repository was assembled from two working copies of the THiRST scripts (`3DST_pipeline_1/Registration_new`
and `3DST_pipeline_2`), each run on a different tissue block. Cross-checking the two copies showed
the same algorithms in both; the differences were machine paths, case identifiers, the class name of the
region annotation (`sample` vs `normal`), section numbering (2-13 vs 1-12) and Space Ranger file
locations, all of which now live in the YAML config. Copy 1 contributed the global registration steps
(Steps 1-3) that copy 2 did not contain; copy 2 contributed the later analysis and figure scripts.

## Original script → this repository

| Original | Now |
|---|---|
| `draw_sep_svs_point_center_fast.py` | `scripts/0_preprocessing/export_st_centers.py` (+ `--save-png`, the commented-out image export) |
| `draw_sep_svs_point_contour_fast_csv.py` | `scripts/0_preprocessing/export_st_contours.py` |
| `Step1_affine_registration_2d_PAS_2048_xfeat.py`, `Step5_..._patchlevel.py` | `scripts/1_registration/xfeat_affine.py --level global\|patch` |
| `Step2_xFeat_matrix_nptomat.py`, `Step6_..._patchlevel.py` | `scripts/1_registration/affine_to_itk.py --level ...` |
| `Step2_superglue+ANTs.py`, `Step7_xfeat+ANTs_patchlevel.py` | `scripts/1_registration/ants_affine.py --level ...` |
| `superglue_ants_registration_onpair_new_RGB.py` (`register_a_pair`) | `thirst/ants.py` |
| `Step3_registertoMiddle_ANTs.py`, `Step8_registertoMiddle_ANTs_patchlevel.py` | `scripts/1_registration/warp_to_reference.py --level ...` |
| `sampling_spot.py` | `scripts/1_registration/sample_patch_centers.py` |
| `Step4_get_3D_Patch_with_spotlocation_crop3Dimages.py` | `scripts/1_registration/crop_patch_stacks.py` |
| `Step3.5_registertoMiddle_ANTs_points_center_fast.py` | `scripts/2_coordinate_transfer/global_spot_centers.py` |
| `Step3.5_registertoMiddle_ANTs_points_contour_fast.py` | `scripts/2_coordinate_transfer/global_spot_contours.py` |
| `Step3.5_registertoMiddle_ANTs_points_object_fast_ori.py` | `scripts/2_coordinate_transfer/global_objects.py` |
| `Step8.5_registertoMiddle_ANTs_points_center_patchlevel_fast.py` | `scripts/2_coordinate_transfer/patch_spot_centers.py` |
| `Step8.5_registertoMiddle_ANTs_points_contour_patchlevel_fast.py` | `scripts/2_coordinate_transfer/patch_spot_contours.py` |
| `Step8.5_registertoMiddle_ANTs_points_object_patchlevel_fast_new_generegions_ori.py` | `scripts/2_coordinate_transfer/patch_objects.py` |
| `Step9.5_contourMatching_parellel_json_patch_reg.py` | `scripts/3_st_association/v2_v2_iou.py` |
| `Step9.7_contourMatching_parellel_json_ST_to_HD_sep_usedistance_patchreg.py` | `scripts/3_st_association/v2_hd_iou.py` |
| `Verify_IoU_json_merge.py` | `scripts/3_st_association/merge_v2_hd_iou.py` |
| `Step11_Match_Object_tracking_results_QuPath_allglom_0915.py`, `..._onlylongglom_0915.py` | `scripts/4_object_tracking/track_objects.py` (`--min-track-len 10` = "only long glomeruli") |
| `Step9_contourMatching_object_new.py`, `Step9_contourMatching_object_global.py` | `scripts/4_object_tracking/track_objects_greedy.py --registration patch\|global` |
| `Step12_Match_Object_tracking_results_itemcheck.py` | `scripts/4_object_tracking/check_track_counts.py` |
| `Area_cal.py` | `scripts/4_object_tracking/cross_section_area.py` |
| `Step10_Match_Object_Spot_patchregistration_using_patch.py` | `scripts/5_object_analysis/match_objects_to_spots.py` |
| `Step10.5_Match_Object_gene_patchregistration_using_patch_gene_findbarcode.py` | `scripts/5_object_analysis/export_track_barcodes.py` |
| `Step10_Match_Object_gene_patchregistration_using_patch_gene_highselect_colorbar_withYanglist.py` | `scripts/5_object_analysis/gene_maps_per_track.py` |
| `Step10_..._colorbar_withYuclustering_cell_bbox_QuPath_0813.py` | `scripts/5_object_analysis/qupath_cells_by_cluster.py` |
| `Step10_..._colorbar_withYuclustering_allcell_bbox_QuPath_0911.py` | `scripts/5_object_analysis/qupath_all_cells_by_cluster.py` |
| `Figure_results_code_Resigration_results_chessfigure_{global,patch,patch_after}.py` | `scripts/6_visualization/checkerboard.py --mode global\|patch-before\|patch-after` |
| `Figure_results_code_Trackingfigure.py` | `scripts/6_visualization/tracking_figure.py` |
| `Step12_3D_sample_visualization_whitebackground.py` | `scripts/6_visualization/render_3d_objects.py` |
| `Verify_pointlocation_local_contour_Ori_only_sample.py` | `scripts/6_visualization/qc_spot_sampling.py` |
| `test.mat` | `thirst/assets/itk_affine_template.mat` |

### Not carried over

| Original | Reason |
|---|---|
| `Step3.5_..._object_fast.py`, `Step8.5_..._object_patchlevel_fast_new.py`, `Step8.5_..._fast_new_generegions.py` | superseded by the `_ori` versions (same output + coordinate log); `_fast_new` stopped on contours leaving a patch |
| `Step9_contourMatching_object_new_connectglom.py`, `..._withcenter_check.py`, `..._withori.py` | earlier versions of the Hungarian tracker now in `track_objects.py` |
| `Step11_Match_Object_tracking_results_QuPath_longglom_debugging.py` | debugging variant |
| `Step10_Match_Object_Spot_patchregistration_using_global.py` | comparison variant of `match_objects_to_spots.py` on global registration only |
| `Step10_..._colorbar_withYanglist_cell_bbox_QuPath_0813.py` | did not run (tab/space indentation error; expression matrix loading commented out) |
| unused functions of `superglue_ants_registration_onpair_new_RGB.py` (`register_3D`, `return_bbox`, `forward_bbox`, `register_a_pair_back`, ...) | not used by the pipeline |
| `image_7_..._center_sample.csv` | derived data of one case; regenerated by `sample_patch_centers.py` |
| `3DST_pipeline_1/Omni-Seg/` | empty in the upload |

### Which tracks each analysis read

Two trackers write 3D objects: `track_objects.py` (Hungarian assignment; originally the
`Step11_..._QuPath_*_0915.py` scripts, folder `Object_IoU_patch_connectglom`) and
`track_objects_greedy.py` (greedy best overlap; `Step9_contourMatching_object_new.py`, folder
`Object_IoU_patch`). The original analysis scripts had the track folder written into their code, and the
two working copies do not agree on it everywhere:

| This repository (original script) | Copy 1 read | Copy 2 read | Default here |
|---|---|---|---|
| `cross_section_area.py` (`Area_cal.py`) | Hungarian | greedy | greedy |
| `match_objects_to_spots.py` (`Step10_Match_Object_Spot_patchregistration_using_patch.py`) | greedy | Hungarian | Hungarian |
| `export_track_barcodes.py` (`Step10.5_..._findbarcode.py`) | — | Hungarian | Hungarian |
| `gene_maps_per_track.py` (`Step10_..._highselect_colorbar_withYanglist.py`) | greedy | greedy | greedy |
| `qupath_cells_by_cluster.py` (`Step10_..._withYuclustering_cell_bbox_QuPath_0813.py`) | — | greedy | greedy |
| `tracking_figure.py` (`Figure_results_code_Trackingfigure.py`) | — | Hungarian | Hungarian |
| `render_3d_objects.py` (`Step12_3D_sample_visualization_whitebackground.py`) | — | greedy | greedy |
| `check_track_counts.py` (`Step12_Match_Object_tracking_results_itemcheck.py`) | Hungarian | Hungarian | Hungarian |

(— : not in that copy.) The code as last saved does not show which tracker produced a published result:
it may have been edited or run differently in between, and the two copies differ in the first two rows.
The original outputs can show it: `Area_cal.py` wrote its tables into the folder of the tracks it read;
the object–spot matching wrote `Object_tracking_IoU_affine_patch_connectglom` from Hungarian tracks and
`Object_tracking_IoU_affine_patch` from greedy tracks; the barcode export and the tracking figure wrote
folders whose names end in `_connectglom` (or `_connectglom_global_vis`). For the other outputs,
compare the objects in a figure (track IDs, number of sections per object) with the two
`IoU_Object.json` files. In this repository the output folder names always say which tracks were read
(see *Fixes after review*).

## Changes made while organising

Mechanical (checked by comparing syntax trees): removed unused imports, `sys.path` hacks pointing to
`airlab`, comment lines with machine paths, and functions that were never called; tabs → spaces.

Interface:
* all paths, case identifiers and dataset-specific names come from the YAML config; the case name is
  taken from the case folder (e.g. GeoJSON `<case>-<section>.svs.geojson`, WSI `<case>-<section>.svs`);
* the reference section and the reference spot table are derived from the stack
  (`int(len(images) / 2)`) instead of the hard-coded `6` / `image_7_..._center.csv`;
* new options for analyses that were previously switched by editing code: `--level`, `--registration`,
  `--tracks-subdir`, `--min-track-len`, `--min-distance`, `--save-png`.

Fixes that keep the original results:
* `antsRegistrationSyN.sh ... - o <prefix>` → `-o <prefix>`: with the stray space ANTs ignored the
  prefix and wrote `output*` into the working directory, which was the intended folder anyway;
* `new_data['col'][idx] = v` → `new_data.loc[idx, 'col'] = v` in `patch_spot_centers.py` and
  `patch_spot_contours.py`: with pandas ≥ 3 the chained assignment silently writes nothing and every
  patch-refined coordinate stays 0; the patch-name columns are created with `object` dtype;
* `shapely.vectorized.contains` (deprecated) → `shapely.contains_xy`.

## Fixes after review

An external review reproduced failures that both original implementations share, so the agreement
between the two implementations and with this repository said nothing about them. Each is fixed and
covered by a test in [`tests/`](../tests) (or by a demo variant). On the synthetic regression data
(see Validation) the outputs do not change, except for the format of the object files and the QuPath
per-track export, which now works. That data does not show that the published results are unaffected:
it has float coordinates, one object class, no failed registration and no empty section.

| Original behaviour | Now | Covered by |
|---|---|---|
| `ants_affine.py` printed a warning when ANTs failed and continued; `global_objects.py` and `patch_objects.py` skipped a missing pair of a section-to-reference chain and applied the others, so coordinates were silently wrong | ANTs failures (non-zero exit status, or no readable `output0GenericAffine.mat`) are collected and the script exits with status 1; every script that chains transforms first checks that all pairs of the stack (and of every patch stack) are present and readable, before writing anything (`thirst/transforms.py`) | `test_registration.py` |
| A restart reused whatever `output0GenericAffine.mat` was present, including one written by an ANTs run that then failed or was interrupted; a partly written NIfTI from `ConvertImagePixelType` was also reused | a pair being (re)computed starts from an empty `step2_run_ants_reg/`; on failure its contents are moved to `step2_run_ants_reg.failed/`; only a successful run writes `ants_completed.json` (command, ANTs version, SHA-256 of the affine), and a restart skips a pair only if that record matches the affine; other existing affines are recomputed unless `--trust-existing` is given; NIfTI conversions are written under a temporary name | `test_registration.py` (ANTs stand-in that writes a valid affine and then exits with status 42) |
| `xfeat_affine.py`: a pair with ≤ 10 inlier matches silently got an identity initialisation; a pair without any affine made `affine_to_itk.py` crash later; an initialisation left by an earlier run could be picked up | status of every pair in `registration_xfeat/xfeat_status.csv` (`ok` / `identity_fallback` / `failed`), summary at the end, exit status 1 if a pair failed; earlier initialisations are removed before each pair; clear error in `affine_to_itk.py` | manual run (needs torch) |
| `track_objects.py` dropped sections without objects from its section list and then looped over `range(1, number of non-empty sections)`: with *k* empty sections the last *k* sections were never tracked | every section is visited (`track_sections`); empty sections are named in a warning | `test_tracking.py` |
| Object GeoJSONs were listed with a glob, so a missing file shifted the slice index of every later section (wrong section ↔ WSI mapping in the tracker and in stages 5–6) | one GeoJSON per section image, in stack order; a missing file is an error | `test_tracking.py` |
| `track_objects_greedy.py` crashed on a section without the object class; its matcher blocked a contour of a second class that had the same per-class index | handled | `test_tracking.py` |
| `patch_spot_contours.py` chose V2 or HD from the number of rows (`> 10000` → HD): an HD region with ≤ 10,000 bins crashed (`KeyError: now_x_4`), and a V2 capture area with > 10,000 spots would have been read as 4-corner squares | chosen from the polygon columns (100 vertices = V2 spot, 4 = HD bin), also in `global_spot_contours.py` | `test_inputs_and_tables.py`; demo with 7,500-bin HD regions |
| `v2_v2_iou.py` and `v2_hd_iou.py` looked barcodes up by label in a filtered Series: when units without coordinates (`now_x == 0`) were left out, IoUs were filed under the barcodes of other units, or `KeyError` was raised | barcodes are taken by position, in the order of the polygons; the centre and polygon tables must list the same barcodes in the same order | `test_st_association.py` |
| `v2_v2_iou.py` crashed on an empty batch (a section without units with coordinates, or exactly *n* × 1,000 of them) | batch count `ceil(n / 1000)` | `test_inputs_and_tables.py` |
| `v2_hd_iou.py` and `merge_v2_hd_iou.py` indexed past the last section when the stack has an odd number of sections; the merge pattern `row_<i>_col_<j>*` also matched the files of columns `<j>0`–`<j>9` | HD loop `range(1, N, 2)`; pattern `row_<i>_col_<j>_*.json` | `test_st_association.py` |
| The V2/HD alternation assumed by stages 3, 5 and 6 was not checked | checked against `st_sections` (`require_alternating_platforms`) | `test_st_association.py`, `test_inputs_and_tables.py` |
| Stage 5–6 scripts look a tracked contour up by its index within its class, which is ambiguous with several classes | exactly one entry in `object_classes` is required there | `test_inputs_and_tables.py` |
| `export_st_*.py` applied the `in_tissue` flag of `in_tissue_positions` by row position | matched by barcode | `test_inputs_and_tables.py`; demo with shuffled flag tables |
| `track_objects.py` (Hungarian) linked objects of different classes; the documented `--min-track-len` output (`IoU_Object_longglom.json`) was never read by the stage 4–6 scripts, which read `IoU_Object.json` (possibly stale) | only same-class objects are linked (also in the greedy tracker); `--tracks-suffix` selects the track file in every consumer, whose outputs then carry the suffix; a missing file is an error | `test_objects_and_tracks.py` |
| `qupath_cells_by_cluster.py` built cluster keys from short HD barcodes but looked them up with the Space Ranger form (`s_008um_…`), so no cell was ever annotated on standard data | both sides normalised with `thirst.st.short_hd_barcode` (also used by the all-cells script) | `test_objects_and_tracks.py`; regression data (see Validation) |
| Stage 2 object files used a non-standard `Polygon` nesting (one feature per class holding all contours) that GIS tools and QuPath cannot read, and `patch_objects.py` repeated the growing contour list in every feature (file size quadratic in the number of objects) | standard GeoJSON, one `Polygon` per object with `contour_id`; one shared reader (`thirst/objects.py`) for all scripts, which also reads the earlier layout and `MultiPolygon` inputs | `test_objects_and_tracks.py`, `test_registration.py` |
| `global_objects.py` truncated integer input coordinates to whole 4x pixels (`int` array assignment) | coordinates are read as floats | `test_registration.py` |
| `*_spot_contours.py` trusted the polygon table alone; `gene_maps_per_track.py` and `qupath_cells_by_cluster.py` mapped slice indices to sample names in `st_sections` order, wrong if the config lists other sections | the polygon type must agree with `st_sections`; sample names are taken in stack order | `test_st_association.py`, `test_objects_and_tracks.py` |
| `demo/check_demo.py` passed with no case or missing files (`mean([])` = NaN never exceeds a threshold); the simulated demo used identity patch transforms, so patch-level refinement was never exercised | every expected case, section file, track and IoU table is required and errors must be finite; the simulated global registration carries a known residual that the (exact, non-identity) patch transforms must remove; the registration mode is reported and, for `--real`, the XFeat status files and ANTs warp fields must exist | `test_objects_and_tracks.py`; demo |
| XFeat imports `tqdm`, which was not a requirement; torch.hub pins neither XFeat's code nor its weights | `tqdm` added; `xfeat_hub` takes `owner/repo:<commit>` (accepted although torch.hub only validates branch and tag heads) or a local clone; `xfeat_weights_sha256` stops the run if the weights differ; `xfeat_provenance_<level>.json` records the ref, the git commit when known, SHA-256 of the code files and of the weights, and the torch version; a first-use trust prompt in a non-interactive run gives a clear error | `test_xfeat_status.py` |
| Stage 5–6 scripts wrote to fixed folder names whatever `--tracks-subdir` said: the object–spot matching, the barcode export and the tracking figure wrote to `…_connectglom` folders (the Hungarian tag) also when they read greedy tracks, overwriting results of the Hungarian tracks; the gene maps, 3D renderings and per-track QuPath export had the reverse problem | output folder names carry the tag of the tracks read (`_connectglom` = Hungarian, none = greedy, `_<folder>` otherwise), the convention of the original scripts (copy 1 wrote `Object_tracking_IoU_affine_patch` from greedy tracks, copy 2 `…_patch_connectglom` from Hungarian tracks); names for the default tracks are unchanged | `test_objects_and_tracks.py`; regression data (see Validation) |
| The XFeat revision was not fixed by any configuration | `configs/example.yaml` and the demo use the release configuration (see *XFeat version*); `check_demo.py` requires verified weights in `--real` runs and reports the XFeat commit | `test_xfeat_status.py`; `--real` demo |
| `cross_section_area.py` raised `KeyError: 'track_id'` on a track file without tracks (e.g. none reaching `--min-track-len`) | empty tables with headers; every track consumer reports an empty track file (all were run on one) | `test_tracking.py` |
| `global_objects.py` / `patch_objects.py` printed and ignored an error writing `contours_before_after.csv` (needed by the tracker); `patch_objects.py` left out contours outside the window of their nearest patch stack without a summary | a write error stops the script; left-out contours are counted per section | — |
| `demo/run_demo.sh DIR --real` after a simulated run in `DIR` reused the ground-truth transforms, because `ants_affine.py` skips pairs that already have a result | each mode has its own directory (default `demo_data` / `demo_data_real`), recorded in `DIR/.demo_mode`; mixing modes is refused | `test_registration.py` |

Results computed with the original scripts can differ from this version wherever one of these
conditions occurred, and should be recomputed or checked there:

* object annotations with integer coordinates (typical of mask-derived polygons): object coordinates
  are no longer truncated to whole 4x pixels (shifts of up to 1 px at 4x);
* more than one class in `object_classes`: objects of different classes are no longer linked;
* Space Ranger HD barcodes in the spot tables: `qupath_cells_by_cluster.py` now finds the clusters;
* a failed, missing or partial registration pair, a section without objects, HD regions of ≤ 10,000
  bins, units without coordinates, or an `in_tissue_positions` table in another row order: the
  original scripts gave wrong or incomplete results (or stopped) there.

Which of these apply to the data of the paper has not been checked here; the paper's tissue data
were not available for this release. `v2_hd_iou.py` keeps batch files that already exist and
`ants_affine.py --trust-existing` keeps transforms without a completion record, so delete earlier
outputs before recomputing them with this version.

Unchanged on purpose, because it determines the published results: only the affine part of the ANTs
registration is applied (the SyN warp fields are computed but not used), at the global and at the
patch level. See the README's *Status, scope and known limitations*.

## XFeat version

THiRST loads XFeat through torch.hub (`hubconf.py` of `verlab/accelerated_features`) and calls
`XFeat.match_xfeat(..., top_k=4096)` on a (1, 1, H, W) tensor of the grayscale section image, resized so
that its longer side is 1,000 px (`sg_affine` in `scripts/1_registration/xfeat_affine.py`).

**Release configuration** (`thirst/provenance.py`, `configs/example.yaml`, demo): commit
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a` (head of the main branch on 2026-11-09, committed
2025-01-15) and the weights that its torch.hub entry point downloads, `weights/xfeat.pt` of the main
branch, SHA-256 `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`. The code digest that
`xfeat_provenance_<level>.json` records for this commit (`401ae2a794771f91…`) is also the digest that an
independent review recorded for a torch.hub download of the default branch.

**Revisions since the release.** The weights file has not changed since the release commit `9539f1f`
(2024-04-29). The code that THiRST runs changed in three commits, none of which changes the result of
THiRST's call:

* `65ae30b` (2024-06-22) made the keypoint detection threshold a parameter whose default is the earlier
  fixed value, 0.05;
* `8728d2f` (2024-07-21) added the optional LightGlue matcher (not used by THiRST) and a check whether
  `kornia` can be imported;
* `8728d2f` and `515c4ce` (2024-09-29) changed how `preprocess_tensor` converts numpy arrays; THiRST
  passes a torch tensor, which every revision handles in the same way.

`match_xfeat`, `match`, `parse_input`, `NMS`, `get_kpts_heatmap`, `modules/model.py` and
`modules/interpolator.py` are unchanged since the release (compared by syntax tree, docstrings
ignored). On three section pairs of the demo, with THiRST's preprocessing (CPU, torch 2.14.1), the
release commit and the pinned commit gave bit-identical matches (1,078, 1,188 and 1,073 keypoint
pairs).

**Published results.** The XFeat revision used for the paper was not recorded. Any copy of the
repository downloaded after its release runs the same XFeat computation for THiRST's call, provided
the default weights were used, and that is the open point: the code folder of an old torch.hub cache
does not show which weights were used. The cached `checkpoints/xfeat.pt` on the machine that ran the
registration does; compare its SHA-256 with the release configuration before stating that the
published registration used the same XFeat. Identical code and weights can still give slightly
different keypoints with another torch version, or on a GPU instead of a CPU.

## Validation

**Equivalence with the original scripts on valid input.** The original scripts (unchanged, pandas 2.2;
archive `3DST_pipeline.zip`, SHA-256 `30942f02937799fe004d67a6e11bf35f615c057ab3794170fdeef3bb5c8105b3`) and the scripts in this repository were run on the same synthetic
12-section block (a test harness separate from [`demo/`](../demo): generated WSIs, region and object
GeoJSONs, Space Ranger tables, clustering and expression tables) with identical inputs and simulated
XFeat/ANTs results. After the review fixes the repository scripts were run again from scratch (pandas
3.0) and every output was compared with those of the original scripts — CSV/JSON byte-identical, images
pixel-identical, transforms parameter-identical:

| Outputs | Files compared | Identical |
|---|---|---|
| `image_root`: section images, XFeat (SIFT stand-in) and ANTs (stand-in using the real option parser) results, ITK transforms, registered stacks, patch stacks, checkerboards, QC images | 1,520 | 1,520 |
| `point_root`: ST unit centres (raw, global, patch-refined) and patch-centre sample | 37 | 37 |
| `contour_root`: ST unit polygons, V2-V2 and V2-HD IoU tables, before/after CSVs, 3D tracks, QuPath exports, object-level analyses and figures | 636 | 636 |
| `contour_root`: object files of stage 2 (`annotation_QA*/`, new standard GeoJSON layout) | 24 | 24 with identical contours (70 per stage, same order and coordinates) |
| all-cells QuPath annotations | 6 | 6 |
| **Total** | **2,223** | **2,223** |

Known differences: the 24 stage-2 object files now use standard GeoJSON (compared contour by contour;
the patch-level file of section 1 shrinks from 200 kB to 38 kB because contours are no longer
repeated); `qupath_cells_by_cluster.py` now writes 35 files with 1,225 cell boxes where the original
wrote none (the regression data use Space Ranger HD barcodes, see *Fixes after review*); new files
`xfeat_status.csv` and `xfeat_provenance_global.json`. The first comparison, before the review,
counted 2,234 files; the difference is 11 copies of simulated transforms that the harness had kept,
which are not pipeline outputs. On the same data the original patch-level transfer run with pandas 3
exits without error but returns all-zero coordinates, which is why the `.loc` change above matters.

This shows that the reorganised code behaves like the original on input of this kind. It does not
show that the methods are right in every situation: the failures listed under *Fixes after review*
were shared by both original implementations and did not occur in this data. The original scripts are
not part of this repository, so the comparison cannot be repeated from it alone.

**Tests of the failure modes.** `python -m pytest tests` runs 55 tests in about 30 s (pandas 2.2,
pandas 3.0, a clean environment installed from `requirements.txt` with pandas 3.1, and a conda-forge
environment with CPU PyTorch 2.14 and ANTs 2.6.5). Each builds a small input for one problem in the table above;
with the scripts of the first release (before the review) and the current shared library, 25 of them
fail (several fixes are in the shared library itself, which those scripts then also use).

**Demo and variants.** The demo's simulated global registration is off by a known 2–6 px per section
and the patch-level transforms are the exact corrections. The standard demo (run in the clean
environment) gives 12.01 px before registration, 5.14 px after the global stage (0.47 px from the
expected residual position, i.e. integer rounding) and 0.46 px after patch-level refinement, with all
six objects and a median best V2–V2 IoU of 0.964. With the patch-level transforms replaced by
identity — the earlier demo — the check now fails (5.14 px after "refinement"). Variants: HD regions
cut to about 7,300 bins per section with the `in_tissue` flags in a separately shuffled table
(exported bins follow the flags; the version before the review stopped with `KeyError: 'now_y_4'` in
`patch_spot_contours.py` and exported 3,790 out-of-tissue bins of 7,378 on section 2); sections numbered 2–13 (same result as the standard demo);
and the `--real` path with stand-ins for XFeat (SIFT) and ANTs (returns the initialisation), which
recovers the synthetic transforms to 0.11 px and passes the check that both steps ran for all five
stacks. A `--real` run with the actual tools (XFeat at the release commit, ANTs 2.6.5) has not been
completed yet; an independent reviewer ran the actual XFeat model on one synthetic section pair
(status `ok`), and the comparison under *XFeat version* ran it on three. After the restart and empty-track fixes the ANTs stand-in
step was rerun on the regression data: the registration outputs are unchanged apart from the new
`ants_completed.json` records, and the area tables are byte-identical.
