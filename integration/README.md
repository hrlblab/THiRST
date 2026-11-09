# Cross-technology integration (Visium V2 + Visium HD) — not yet included

Visium V2 and Visium HD sections are integrated in Seurat (Methods 4.3): log-normalisation, shared
highly variable genes, PCA, anchor-based canonical correlation analysis (CCA), joint clustering and UMAP,
followed by batch-mixing entropy / LISI to quantify integration. The resulting cluster labels are
mapped back onto the 3D coordinates produced by this pipeline.

The scripts for this stage will be added here. They connect to the pipeline through:

* `<point_root>/<case>/registration_3D_patch_ANTs/image_*_center.csv` — 3D coordinates of every barcode;
* `<contour_root>/<case>/IoU_ST_to_HD_distance_patchreg_merge/` — V2 spot ↔ HD bin overlaps
  (HD-to-V2 aggregation);
* `.../glomeruli_barcode_per_slice.json` (`scripts/5_object_analysis/export_track_barcodes.py`) —
  HD barcodes of every 3D object for object-level differential expression;
* a meta table with columns `Sample`, `Barcode`, `seurat_cluster.projected`
  (`analysis.clustering_meta_csv`) used by the QuPath cell-annotation scripts.
