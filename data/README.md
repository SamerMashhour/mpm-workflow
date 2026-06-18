# Data policy

This repository intentionally contains **no real workshop, proprietary, or third-party source data**.

It provides two original, deterministic synthetic MPM scenarios through `mpm_workflow.synthetic` and the `mpm-workflow generate-synthetic` command:

- `belt_cover`: broad arcuate mineral-system belts, volcanic and intrusive complexes, and cover basins.
- `structural_corridors`: intersecting structural corridors, relay structures, and compact mineral-system clusters.

Generate each 30,000-cell input locally:

```bash
mpm-workflow generate-synthetic --scenario belt_cover --output data/generated_belt_cover.csv
mpm-workflow generate-synthetic --scenario structural_corridors --output data/generated_structural_corridors.csv
```

Both outputs share the same 33-column input schema. Their H3-style IDs are deliberately synthetic and are not valid H3 identifiers. Coordinates, geological domains, covariate values, missing-data patterns, anomalies, labels, and target geometries are newly simulated.

Do not add real datasets here unless you have explicit permission to publish them.
