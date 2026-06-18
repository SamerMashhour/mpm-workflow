# Synthetic input data

`mpm-workflow` generates two fully synthetic, openly shareable examples. Both use the same **input schema family** as the MPM workflow: H3-style cell IDs, spatial coordinates, sparse positive annotations, labelled background cells, categorical geology, geochronology, signed geophysics, incomplete radiometrics, and multi-element lake-sediment geochemistry.

| Case | Geological-style architecture |
|---|---|
| `belt_cover` | Broad curved belts, volcanic and intrusive complexes, cover basins, and dispersed target clusters. |
| `structural_corridors` | Intersecting fault corridors, relay structures, compact clusters at structural intersections, and separate cover domains. |

Neither generated CSV is a copy, subset, transformation, resampling, coordinate shift, or perturbation of a workshop dataset. Coordinate envelopes, values, labels, spatial domains, anomaly patterns, and synthetic IDs are produced from scratch using separate random seeds and separate latent geological architectures.

## What the generator does

Each scenario starts from a staggered H3-style lattice. A scenario-specific latent architecture controls broad basement domains, lithology, geochronology, signed magnetic and gravity fields, partially surveyed radiometrics, and lake-sediment geochemistry. Sparse positive annotations are then selected from a noisy latent mineral-system score.

The latent score is not exported. It is used only to construct simulated labels. Model performance therefore shows that the package can learn a deliberately simulated signal. It is not a geological finding, mineral potential map, or validation result for any real region.

## H3 identifiers

`H3_ADDRESS` values such as `synthetic_belt_cover_r7_000001` are unique placeholders, not valid H3 indices. The workflow needs only unique cell IDs and coordinate columns. Replace them with genuine H3 indices when applying the package to a real data cube.

## Regeneration

Generate both full-size cases and their local workflow artifacts with:

```bash
pip install -e .[viz]
python examples/generate_and_run_synthetic_cases.py
```

Generate either architecture independently:

```bash
mpm-workflow generate-synthetic --scenario belt_cover --output data/generated_belt_cover.csv
mpm-workflow generate-synthetic --scenario structural_corridors --output data/generated_structural_corridors.csv
```
