"""Original synthetic data generators for the MPM workflow.

The two scenarios are deliberately simulated. They do not re-use coordinates,
values, labels, or spatial patterns from the workshop input data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticMPMConfig:
    """Parameters for a reproducible, H3-style synthetic MPM data cube."""

    n_cells: int = 30_000
    h3_resolution: int = 7
    positive_count: int = 760
    negative_training_count: int = 760
    random_state: int = 20260618
    label_noise: float = 0.72
    scenario: str = "belt_cover"
    latitude_range: tuple[float, float] = (48.0, 51.0)
    longitude_range: tuple[float, float] = (-115.5, -107.5)


def _distance_to_polyline(
    x: np.ndarray, y: np.ndarray, points: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Distance from points to a piecewise-linear trace in normalized map space."""
    vertices = np.asarray(points, dtype=float)
    distance2 = np.full(x.size, np.inf)
    for start, end in zip(vertices[:-1], vertices[1:]):
        dx, dy = end - start
        length2 = dx * dx + dy * dy
        projection = ((x - start[0]) * dx + (y - start[1]) * dy) / length2
        projection = np.clip(projection, 0.0, 1.0)
        nx = start[0] + projection * dx
        ny = start[1] + projection * dy
        distance2 = np.minimum(distance2, (x - nx) ** 2 + (y - ny) ** 2)
    return np.sqrt(distance2)


def _ribbon(x: np.ndarray, y: np.ndarray, points: Sequence[tuple[float, float]], width: float, amplitude: float = 1.0) -> np.ndarray:
    distance = _distance_to_polyline(x, y, points)
    return amplitude * np.exp(-(distance**2) / (2.0 * width**2))


def _ellipse(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    major: float,
    minor: float,
    angle_degrees: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    dx, dy = x - x0, y - y0
    u = np.cos(angle) * dx + np.sin(angle) * dy
    v = -np.sin(angle) * dx + np.cos(angle) * dy
    return amplitude * np.exp(-0.5 * ((u / major) ** 2 + (v / minor) ** 2))


def _texture(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, count: int = 36) -> np.ndarray:
    texture = 0.06 * np.sin(6.0 * np.pi * x + 0.5) * np.cos(4.3 * np.pi * y - 0.4)
    for _ in range(count):
        texture += _ellipse(
            x,
            y,
            float(rng.uniform(0.02, 0.98)),
            float(rng.uniform(0.02, 0.98)),
            float(rng.uniform(0.025, 0.11)),
            float(rng.uniform(0.018, 0.07)),
            float(rng.uniform(0, 180)),
            float(rng.normal(0.0, 0.06)),
        )
    return texture


def _architecture(u: np.ndarray, v: np.ndarray, rng: np.random.Generator, scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two intentionally different latent map architectures."""
    if scenario == "belt_cover":
        belt_a = _ribbon(u, v, [(0.06, 0.23), (0.15, 0.34), (0.24, 0.48), (0.29, 0.64), (0.38, 0.76)], 0.038, 1.25)
        belt_b = _ribbon(u, v, [(0.25, 0.24), (0.38, 0.26), (0.50, 0.36), (0.60, 0.52), (0.64, 0.65)], 0.044, 1.13)
        belt_c = _ribbon(u, v, [(0.50, 0.82), (0.61, 0.74), (0.73, 0.72), (0.84, 0.78), (0.94, 0.88)], 0.036, 1.08)
        structure = belt_a + belt_b + belt_c
        volcanic = _ellipse(u, v, 0.28, 0.64, 0.105, 0.072, 28, 1.10) + _ellipse(u, v, 0.57, 0.44, 0.13, 0.065, -35, 0.95) + _ellipse(u, v, 0.82, 0.77, 0.11, 0.055, 20, 0.88)
        intrusive = _ellipse(u, v, 0.17, 0.51, 0.07, 0.045, -15, 0.90) + _ellipse(u, v, 0.48, 0.22, 0.09, 0.055, 35, 0.98)
        cover = _ellipse(u, v, 0.68, 0.52, 0.18, 0.13, -20, 1.10) + _ellipse(u, v, 0.41, 0.87, 0.16, 0.08, 12, 0.60)
        basement = 0.38 * np.sin(2.5 * np.pi * u + 0.6) + 0.31 * np.cos(2.0 * np.pi * v - 0.5)
    elif scenario == "structural_corridors":
        fault_a = _ribbon(u, v, [(0.05, 0.84), (0.17, 0.73), (0.31, 0.62), (0.43, 0.48), (0.57, 0.36), (0.69, 0.23), (0.82, 0.12)], 0.025, 1.22)
        fault_b = _ribbon(u, v, [(0.12, 0.12), (0.23, 0.28), (0.34, 0.42), (0.46, 0.56), (0.58, 0.70), (0.69, 0.84), (0.86, 0.94)], 0.027, 1.15)
        relay = _ribbon(u, v, [(0.22, 0.48), (0.34, 0.46), (0.48, 0.48), (0.59, 0.53), (0.72, 0.61)], 0.022, 0.95)
        structure = fault_a + fault_b + relay
        volcanic = _ellipse(u, v, 0.38, 0.49, 0.065, 0.043, -30, 1.28) + _ellipse(u, v, 0.60, 0.70, 0.080, 0.045, 35, 1.02) + _ellipse(u, v, 0.76, 0.34, 0.075, 0.048, -22, 0.90)
        intrusive = _ellipse(u, v, 0.33, 0.31, 0.068, 0.036, 42, 0.95) + _ellipse(u, v, 0.50, 0.54, 0.080, 0.042, -12, 0.92)
        cover = _ellipse(u, v, 0.82, 0.76, 0.17, 0.11, -28, 1.05) + _ellipse(u, v, 0.12, 0.42, 0.14, 0.08, 20, 0.72)
        basement = 0.42 * np.sin(3.2 * np.pi * u - 0.45) - 0.26 * np.cos(3.0 * np.pi * v + 0.50)
    else:
        raise ValueError("scenario must be 'belt_cover' or 'structural_corridors'.")
    return structure, volcanic, intrusive, cover, basement + _texture(u, v, rng)


def make_synthetic_mpm(config: SyntheticMPMConfig = SyntheticMPMConfig()) -> pd.DataFrame:
    """Create a simulated 33-column MPM input table, with no real data copied in."""
    if config.n_cells < 100:
        raise ValueError("n_cells must be at least 100.")
    if not 0 < config.positive_count < config.n_cells:
        raise ValueError("positive_count must be between 1 and n_cells - 1.")
    rng = np.random.default_rng(config.random_state)
    n = config.n_cells
    n_cols = int(np.ceil(np.sqrt(n * 2.65)))
    n_rows = int(np.ceil(n / n_cols))
    row, col = np.divmod(np.arange(n), n_cols)
    u = (col + 0.42 * (row % 2) + 0.50 + rng.uniform(-0.18, 0.18, n)) / n_cols
    v = (row + 0.50 + rng.uniform(-0.18, 0.18, n)) / n_rows
    u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
    latitude = config.latitude_range[0] + v * (config.latitude_range[1] - config.latitude_range[0])
    longitude = config.longitude_range[0] + u * (config.longitude_range[1] - config.longitude_range[0])
    structure, volcanic, intrusive, cover, basement = _architecture(u, v, rng, config.scenario)
    latent = 1.20 * structure + 0.95 * volcanic + 0.72 * intrusive + 0.20 * basement - 0.92 * cover
    lith_code = np.select(
        [(volcanic + 0.55 * structure) > 0.78, intrusive > 0.56, cover > 0.64, basement > 0.43, basement < -0.40],
        [4, 1, 3, 5, 2],
        default=0,
    ).astype(int)
    lithologies = np.array([
        "sedimentary_siliciclastic", "igneous_intrusive_intermediate", "igneous_intrusive_felsic",
        "sedimentary_chemical_carbonate", "igneous_extrusive_mafic", "metamorphic_gneiss_orthogneiss",
    ], dtype=object)
    minor_lithologies = np.array([
        "sedimentary_siliciclastic_fine", "igneous_intrusive_felsic", "igneous_extrusive_intermediate",
        "sedimentary_siliciclastic_coarse", "igneous_extrusive_intermediate", "metamorphic_quartzite",
    ], dtype=object)
    lith_majority = lithologies[lith_code]
    lith_minority = lith_majority.copy()
    replacement = rng.random(n) < 0.27
    lith_minority[replacement] = minor_lithologies[lith_code[replacement]]
    age_lookup = np.array([1740, 1825, 1695, 510, 1865, 2250], dtype=float)
    age_majority = np.clip(age_lookup[lith_code] + 70 * basement + rng.normal(0, 42, n), 80, 2600)
    age_minority = np.clip(age_majority + rng.normal(2, 68, n), 80, 2600)
    magnetic_rtf = 210 * volcanic + 160 * intrusive + 125 * basement + 250 * structure + rng.normal(0, 90, n)
    magnetic_1vd = 0.025 * np.sin(5.4 * np.pi * u) + 0.040 * structure + rng.normal(0, 0.040, n)
    magnetic_hgm = 0.025 + 0.105 * structure + 0.055 * volcanic + np.abs(basement) * 0.04 + rng.gamma(1.6, 0.020, n)
    gravity_bouguer = -47 + 8.0 * basement + 4.6 * intrusive - 2.8 * cover + rng.normal(0, 2.8, n)
    gravity_1vd = 0.00028 * np.sin(4.4 * np.pi * v) + 0.00062 * structure - 0.00022 * cover + rng.normal(0, 0.00033, n)
    gravity_hgm = 0.00019 + 0.00046 * structure + 0.00020 * np.abs(basement) + rng.gamma(1.8, 0.00012, n)
    seismic_lab = 205 - 7.0 * volcanic + 3.5 * basement + 3.0 * cover + rng.normal(0, 4.0, n)
    seismic_moho = 44.0 - 0.75 * basement - 0.72 * volcanic + 0.45 * cover + rng.normal(0, 0.50, n)
    available = np.sin(2.2 * np.pi * u - 0.5) + np.cos(2.8 * np.pi * v + 0.4) + 0.55 * rng.normal(0, 1, n) > -1.05
    k = np.where(available, np.clip(0.78 + 0.42 * intrusive + 0.21 * volcanic + rng.normal(0, 0.17, n), 0.03, None), np.nan)
    th = np.where(available, np.clip(3.8 + 1.75 * intrusive + 0.55 * basement + rng.normal(0, 0.72, n), 0.05, None), np.nan)
    uranium = np.where(available, np.clip(0.58 + 0.28 * structure + 0.20 * intrusive + rng.normal(0, 0.14, n), 0.01, None), np.nan)
    signal = np.clip(latent, -1.5, 3.5)
    au = np.exp(1.18 + 0.62 * signal + rng.normal(0, 0.51, n))
    cu = np.exp(3.00 + 0.48 * signal + rng.normal(0, 0.57, n))
    zn = np.exp(4.45 + 0.40 * signal + rng.normal(0, 0.53, n))
    co = np.exp(2.22 + 0.15 * signal + rng.normal(0, 0.22, n))
    ni = np.exp(3.18 + 0.20 * intrusive + 0.08 * structure + rng.normal(0, 0.22, n))
    pb = np.exp(1.60 + 0.19 * signal + rng.normal(0, 0.28, n))
    mo = np.exp(0.72 + 0.17 * signal + rng.normal(0, 0.14, n))
    hg = np.exp(3.88 + 0.14 * signal + rng.normal(0, 0.21, n))
    mn = np.exp(6.18 + 0.20 * basement + rng.normal(0, 0.43, n))
    fe = np.exp(0.92 + 0.12 * volcanic + 0.07 * structure + rng.normal(0, 0.19, n))
    observed_score = latent + 0.21 * np.log1p(au) + 0.15 * np.log1p(cu) + rng.normal(0, config.label_noise, n)
    positive = np.argpartition(observed_score, -config.positive_count)[-config.positive_count:]
    training_class = np.zeros(n, dtype=int)
    training_class[positive] = 1
    deposit_count = max(1, int(round(config.positive_count * 0.087)))
    deposit = positive[np.argsort(observed_score[positive])[-deposit_count:]]
    occurrence = np.setdiff1d(positive, deposit)
    training_deposit = np.zeros(n, dtype=int)
    training_occurrence = np.zeros(n, dtype=int)
    training_deposit[deposit] = 1
    training_occurrence[occurrence] = 1
    training_negative = np.zeros(n, dtype=int)
    background = np.flatnonzero(training_class == 0)
    training_negative[rng.choice(background, size=min(config.negative_training_count, background.size), replace=False)] = 1
    return pd.DataFrame({
        "H3_ADDRESS": [f"synthetic_{config.scenario}_r{config.h3_resolution}_{i:06d}" for i in range(n)],
        "H3_RESOLUTION": config.h3_resolution,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "TRAINING_DEPOSIT": training_deposit,
        "TRAINING_MINERAL_OCCURRENCE": training_occurrence,
        "TRAINING_NEGATIVE": training_negative,
        "TRAINING_CLASS": training_class,
        "GEOLOGY_LITHOLOGY_MAJORITY": lith_majority,
        "GEOLOGY_LITHOLOGY_MINORITY": lith_minority,
        "GEOCHRONOLOGY_MA_MAJORITY": age_majority,
        "GEOCHRONOLOGY_MA_MINORITY": age_minority,
        "GEOPHYSICS_GRAVITY_1VD": gravity_1vd,
        "GEOPHYSICS_GRAVITY_BOUGUER": gravity_bouguer,
        "GEOPHYSICS_GRAVITY_HGM": gravity_hgm,
        "GEOPHYSICS_MAGNETIC_1VD": magnetic_1vd,
        "GEOPHYSICS_MAGNETIC_HGM": magnetic_hgm,
        "GEOPHYSICS_MAGNETIC_RTF": magnetic_rtf,
        "GEOPHYSICS_RADIOMETRIC_K": k,
        "GEOPHYSICS_RADIOMETRIC_TH": th,
        "GEOPHYSICS_RADIOMETRIC_U": uranium,
        "GEOPHYSICS_SEISMIC_LAB": seismic_lab,
        "GEOPHYSICS_SEISMIC_MOHO": seismic_moho,
        "GEOCHEMISTRY_LAKES_PARTIAL_AU": au,
        "GEOCHEMISTRY_LAKES_PARTIAL_CO": co,
        "GEOCHEMISTRY_LAKES_PARTIAL_CU": cu,
        "GEOCHEMISTRY_LAKES_PARTIAL_FE": fe,
        "GEOCHEMISTRY_LAKES_PARTIAL_HG": hg,
        "GEOCHEMISTRY_LAKES_PARTIAL_MN": mn,
        "GEOCHEMISTRY_LAKES_PARTIAL_MO": mo,
        "GEOCHEMISTRY_LAKES_PARTIAL_NI": ni,
        "GEOCHEMISTRY_LAKES_PARTIAL_PB": pb,
        "GEOCHEMISTRY_LAKES_PARTIAL_ZN": zn,
    })


def write_synthetic_mpm(path: str | Path, config: SyntheticMPMConfig = SyntheticMPMConfig()) -> Path:
    """Generate and save one synthetic MPM CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_mpm(config).to_csv(output, index=False)
    return output
