#!/usr/bin/env python3
"""Run the seven-configuration Physics-IQ G/N/M ablation.

The signal file is opened first and all picks are written with a SHA-256
sidecar. Official quality is loaded only after that freeze step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cva_tts import full_factorial_utilities


METHODS = ("G", "N", "M", "G+N", "CVA-Select", "N+M", "G+N+M")
FORMULAS = {
    "G": "z(G)",
    "N": "z(N)",
    "M": "z(M)",
    "G+N": "z(G)+z(N)",
    "CVA-Select": "z(G)+z(M)",
    "N+M": "z(N)+z(M)",
    "G+N+M": "z(G)+z(N)+z(M)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_hashed_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _signal_vector(name: str, values: object, budget: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (budget,) or not np.isfinite(vector).all():
        raise RuntimeError(f"{name} must contain {budget} finite values")
    return vector


def freeze_picks(signals: dict, expected_budget: int) -> dict:
    budget = int(signals.get("candidate_budget", expected_budget))
    if budget != expected_budget:
        raise RuntimeError(f"expected candidate budget {expected_budget}, found {budget}")
    scenes = signals.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("signals must contain a non-empty scenes list")

    frozen_rows = []
    seen_ids: set[str] = set()
    reference_bases: tuple[str, ...] | None = None
    for row in scenes:
        scene_id = str(row["id"])
        family = str(row["family"])
        if scene_id in seen_ids:
            raise RuntimeError(f"duplicate scene id: {scene_id}")
        seen_ids.add(scene_id)
        bases = row.get("bases")
        if not isinstance(bases, dict) or not bases:
            raise RuntimeError(f"{scene_id} has no world-model entries")
        base_names = tuple(sorted(bases))
        if reference_bases is None:
            reference_bases = base_names
        elif base_names != reference_bases:
            raise RuntimeError("world-model coverage differs across scenes")

        base_picks = {}
        for base, components in bases.items():
            g = _signal_vector(f"{scene_id}/{base}/G", components.get("G"), budget)
            n = _signal_vector(f"{scene_id}/{base}/N", components.get("N"), budget)
            m = _signal_vector(f"{scene_id}/{base}/M", components.get("M"), budget)
            utilities = full_factorial_utilities(g, n, m)
            base_picks[base] = {
                method: int(np.argmax(utilities[method])) for method in METHODS
            }
        frozen_rows.append({"id": scene_id, "family": family, "bases": base_picks})

    return {
        "status": "PHYSICS_IQ_PICKS_FROZEN",
        "selected_method": "CVA-Select",
        "configurations": FORMULAS,
        "candidate_budget": budget,
        "selection_uses_official_quality": False,
        "world_models": list(reference_bases or ()),
        "scenes": frozen_rows,
    }


def _quantiles(samples: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def analyze(
    frozen: dict,
    quality: dict,
    *,
    expected_scenes: int,
    expected_families: int,
    expected_bases: tuple[str, ...],
    draws: int,
    seed: int,
) -> dict:
    if frozen.get("selection_uses_official_quality") is not False:
        raise RuntimeError("picks were not frozen behind the quality boundary")
    budget = int(frozen["candidate_budget"])
    quality_rows = quality.get("scenes")
    if not isinstance(quality_rows, list):
        raise RuntimeError("quality must contain a scenes list")
    quality_by_id = {str(row["id"]): row for row in quality_rows}
    if len(quality_by_id) != len(quality_rows):
        raise RuntimeError("quality contains duplicate scene ids")

    measured: dict[str, dict[str, dict[str, list[float]]]] = {
        method: {base: {} for base in expected_bases} for method in METHODS
    }
    for frozen_row in frozen["scenes"]:
        scene_id = frozen_row["id"]
        family = frozen_row["family"]
        if scene_id not in quality_by_id:
            raise RuntimeError(f"official quality is missing scene {scene_id}")
        quality_row = quality_by_id[scene_id]
        if str(quality_row["family"]) != family:
            raise RuntimeError(f"family mismatch for scene {scene_id}")
        if set(frozen_row["bases"]) != set(expected_bases):
            raise RuntimeError(f"unexpected world-model coverage for {scene_id}")
        if set(quality_row["bases"]) != set(expected_bases):
            raise RuntimeError(f"quality world-model coverage changed for {scene_id}")
        for base in expected_bases:
            values = _signal_vector(
                f"{scene_id}/{base}/quality", quality_row["bases"][base], budget
            )
            for method in METHODS:
                pick = int(frozen_row["bases"][base][method])
                measured[method][base].setdefault(family, []).append(float(values[pick]))

    if len(frozen["scenes"]) != expected_scenes:
        raise RuntimeError(
            f"expected {expected_scenes} scenes, found {len(frozen['scenes'])}"
        )
    families = sorted(measured[METHODS[0]][expected_bases[0]])
    if len(families) != expected_families:
        raise RuntimeError(f"expected {expected_families} families, found {len(families)}")

    family_means = {
        method: {
            base: {
                family: float(np.mean(measured[method][base][family]))
                for family in families
            }
            for base in expected_bases
        }
        for method in METHODS
    }
    means = {
        method: {
            "per_world_model": {
                base: float(np.mean(list(family_means[method][base].values())))
                for base in expected_bases
            }
        }
        for method in METHODS
    }
    for method in METHODS:
        means[method]["cross_world_model_mean"] = float(
            np.mean(list(means[method]["per_world_model"].values()))
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(families), size=(draws, len(families)))
    comparisons = {}
    selected = "CVA-Select"
    for method in METHODS:
        if method == selected:
            continue
        differences = np.asarray([
            np.mean([
                family_means[method][base][family]
                - family_means[selected][base][family]
                for base in expected_bases
            ])
            for family in families
        ])
        samples = differences[indices].mean(axis=1)
        comparisons[f"{method}_minus_CVA-Select"] = {
            "delta": float(differences.mean()),
            "ci95": _quantiles(samples),
            "wins": int(np.sum(differences > 1e-12)),
            "ties": int(np.sum(np.abs(differences) <= 1e-12)),
            "losses": int(np.sum(differences < -1e-12)),
        }

    return {
        "status": "PHYSICS_IQ_FULL_FACTORIAL_ABLATION_COMPLETE",
        "evidence_status": "post_hoc_reanalysis_of_frozen_measurements",
        "selected_method": selected,
        "configurations": FORMULAS,
        "population": {
            "scenes": len(frozen["scenes"]),
            "families": len(families),
            "world_models": list(expected_bases),
            "candidate_budget": budget,
        },
        "bootstrap": {"unit": "event_family", "draws": draws, "seed": seed},
        "means": means,
        "paired_family_comparisons": comparisons,
    }


def selftest() -> None:
    bases = ("5B", "a14b", "cosmos")
    signal_rows = []
    quality_rows = []
    for index in range(8):
        scene_id = f"scene-{index:02d}"
        signal_rows.append({
            "id": scene_id,
            "family": f"family-{index:02d}",
            "bases": {
                base: {
                    "G": [0.0, 0.2, 1.0, 0.1],
                    "N": [0.0, 1.0, 0.1, 0.2],
                    "M": [0.0, 0.1, 0.9, 0.2],
                }
                for base in bases
            },
        })
        quality_rows.append({
            "id": scene_id,
            "family": f"family-{index:02d}",
            "bases": {base: [0.0, 0.2, 1.0, 0.1] for base in bases},
        })
    frozen = freeze_picks({"candidate_budget": 4, "scenes": signal_rows}, 4)
    assert all(
        base_picks["CVA-Select"] == 2
        for row in frozen["scenes"]
        for base_picks in row["bases"].values()
    )
    report = analyze(
        frozen,
        {"scenes": quality_rows},
        expected_scenes=8,
        expected_families=8,
        expected_bases=bases,
        draws=1_000,
        seed=7,
    )
    assert report["selected_method"] == "CVA-Select"
    assert len(report["configurations"]) == 7
    print("PHYSICS_IQ_ABLATION_SELFTEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--picks-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-scenes", type=int, default=192)
    parser.add_argument("--expected-families", type=int, default=64)
    parser.add_argument("--expected-bases", default="5B,a14b,cosmos")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not all((args.signals, args.quality, args.picks_output, args.output)):
        parser.error("--signals, --quality, --picks-output, and --output are required")
    if args.budget < 2 or args.draws <= 0:
        parser.error("--budget must be >=2 and --draws must be positive")

    signals = load_json(args.signals)
    frozen = freeze_picks(signals, args.budget)
    frozen["signal_sha256"] = sha256(args.signals)
    picks_sha256 = write_hashed_json(args.picks_output, frozen)

    quality = load_json(args.quality)
    bases = tuple(item.strip() for item in args.expected_bases.split(",") if item.strip())
    report = analyze(
        frozen,
        quality,
        expected_scenes=args.expected_scenes,
        expected_families=args.expected_families,
        expected_bases=bases,
        draws=args.draws,
        seed=args.seed,
    )
    report["inputs"] = {
        "signals_sha256": sha256(args.signals),
        "quality_sha256": sha256(args.quality),
        "frozen_picks_sha256": picks_sha256,
    }
    result_sha256 = write_hashed_json(args.output, report)
    print(
        "PHYSICS_IQ_ABLATION_COMPLETE "
        f"scenes={report['population']['scenes']} sha256={result_sha256}"
    )


if __name__ == "__main__":
    main()
