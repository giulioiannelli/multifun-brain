"""April-batch orchestration scripts.

These are thin (≤80-line) entry points that wire the
:mod:`multifunbrain.datasets.april` loader into
:func:`multifunbrain.pipeline.runner.run_pipeline_batch`.

Order of execution (smoke-test staging):

1. ``00_inventory.py`` — write a manifest CSV.
2. ``01_run_global.py`` — pipeline on 10 global aggregates.
3. ``02_run_bands.py`` — pipeline on 30 band aggregates.
4. ``03_run_patients.py`` — pipeline on 180 per-subject × band files.

Outputs land under
``data/correlation_matrices_results/april/{global,bands,patients}/``
with ``results.pkl`` + ``summary.csv`` + ``config.json`` +
``failed_matrices.json``.
"""
