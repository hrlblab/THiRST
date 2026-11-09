"""THiRST: Three-dimensional High-Resolution Spatial Transcriptomics.

Shared helpers used by the pipeline scripts in ``scripts/``:

* :mod:`thirst.config`     - YAML configuration, case / patch-stack discovery, Space Ranger I/O,
  section-layout checks
* :mod:`thirst.ants`       - ANTs SyN registration of one section pair (XFeat-initialised); raises on failure
* :mod:`thirst.transforms` - checks on the pairwise transforms chained to the reference section
* :mod:`thirst.st`         - helpers and consistency checks for the ST unit tables
"""

__version__ = "0.1.0"
