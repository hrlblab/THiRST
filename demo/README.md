# Demo: synthetic exemplar dataset

```bash
bash demo/run_demo.sh                    # ~5 min on a CPU, no GPU or ANTs needed (data in demo_data/)
bash demo/run_demo.sh --real             # same, but with XFeat (torch.hub) and ANTs (data in demo_data_real/;
                                         # several hours on a 2-core CPU, mostly ANTs)
bash demo/run_demo.sh my_dir [--real]    # any other directory
```

The registration steps keep results that already exist, so the simulated and the real run must not
share a directory: the mode is recorded in `<dir>/.demo_mode` when the data are created, and the script
refuses to run the other mode there (otherwise a `--real` run would silently reuse the ground-truth
transforms).

`make_synthetic_data.py create` writes a 12-section tissue block (Visium V2 on the 1st, 3rd, ...,
Visium HD on the 2nd, 4th, ... section; `--first-section N` numbers them from N) whose sections are
rotated/shifted copies of the middle section, together with QuPath region annotations, a mock
segmentation with six glomerulus-like objects (one misses the first and last section), Space Ranger
tables and a ready-to-use `demo.yaml`.

Without `--real`, the XFeat/ANTs steps are replaced by `make_synthetic_data.py simulate-registration`,
which writes known transforms in the same files and formats. The global transforms are deliberately
off by a small known rotation and shift per section (an imperfect global registration, 2–6 px);
the patch-level transforms are the exact correction of that residual in each patch frame.

`check_demo.py` then requires every expected output and verifies that

* the run used the mode recorded in `.demo_mode`; with `--real`, every global and patch stack has an
  XFeat status file with all pairs `ok` and, for every pair, an ANTs warp field and an ANTs completion
  record (so the tools really ran and finished), and the XFeat weights were verified against the
  SHA-256 of the release configuration (the XFeat commit is reported);
* the V2 spots of every V2 section are about 12 px off the reference grid before registration; after
  the global stage they are off by exactly the simulated residual (simulated) or < 1 px (real); after
  patch-level refinement they are on the grid (< 0.5 px simulated, < 1 px real);
* `track_objects.py` reconstructs the six objects with the expected number of sections;
* every V2 section pair has an IoU table and V2 spots are matched across sections (median best IoU > 0.9).

A missing case, file or result makes the check fail.

With `--real`, `demo.yaml` uses the XFeat release configuration (`xfeat_hub` =
`verlab/accelerated_features:<commit>` and `xfeat_weights_sha256`, from `thirst/provenance.py`), which
torch.hub fetches from GitHub on first use. On a machine without access to GitHub, clone the XFeat
repository, check out that commit, copy its `weights/xfeat.pt` to `$TORCH_HOME/hub/checkpoints/`, and
create the data with `THIRST_XFEAT_HUB=/path/to/accelerated_features bash demo/run_demo.sh DIR --real`.

Every run appends its complete output to `<dir>/demo_run.log` and writes `<dir>/demo_environment.txt`
(command, repository commit, versions of Python, the libraries, torch and ANTs; with `--real` also the
XFeat provenance: source, commit when known, SHA-256 of the code and of the weights). Keep both files
with the results of a `--real` validation run.
