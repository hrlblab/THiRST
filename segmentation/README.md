# Segmentation (Omni-Seg) — not yet included

THiRST segments six renal structures on every WSI with a task- and scale-conditioned
[Omni-Seg](https://github.com/ddrrnn123/Omni-Seg) model (Methods 4.2.1–4.2.4): tissue-guided tiling,
multi-scale sliding-window inference (DT, PT and artery at 10×; capsule and tuft at 5×; PTC at 40×),
whole-slide vote aggregation, and export of the masks to QuPath for expert review.

The code and model weights for this stage will be added here. The rest of the pipeline only needs
its result: one QuPath-compatible GeoJSON per WSI,

```
<object_annotation_dir>/<case>-<section>.svs.geojson
```

with `Polygon` features in full-resolution WSI coordinates and the class in
`properties.classification.name` (e.g. `capsule`); list the classes to reconstruct under
`object_classes` in the config. Any segmentation tool that writes this format can be used.
