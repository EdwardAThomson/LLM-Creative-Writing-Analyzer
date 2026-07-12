"""Benchmarks: frozen manifests (data) plus self-contained benchmark packages (code).

Two kinds of thing live here:

* ``vN.yaml`` manifests: frozen, cumulative metric selections over the shared
  ``utils/metrics`` library (the repetition/fingerprint benchmark). Data, not code;
  resolved by ``utils/metrics/_manifests.py``. Never edited once shipped.
* Benchmark *packages* (currently ``narrative_dynamics/``): a benchmark whose
  object of study is not "N repeated runs of one prompt" but a different unit of
  analysis entirely, packaged as self-contained code with its own manifest series
  (``nd1.yaml``), CLI, and docs. See ``narrative_dynamics/README.md``.

This ``__init__`` deliberately imports nothing, so ``import benchmarks.<pkg>``
stays free of heavy dependencies (the same lazy-import discipline as
``utils/metrics``).
"""
