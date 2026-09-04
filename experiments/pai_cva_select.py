#!/usr/bin/env python3
"""Freeze PAI-Bench-robot picks with the common CVA-Select interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cva_tts import cva_select, within_pool_zscore


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_values(payload: dict, scene: str, budget: int) -> np.ndarray:
    keys = [f"{scene}__c{index:02d}" for index in range(budget)]
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError(f"missing candidates for {scene}: {missing[:3]}")
    values = np.asarray([payload[key] for key in keys], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite values for {scene}")
    return values


def freeze_picks(
    cycle: dict,
    motion: dict,
    consistency: dict,
    i2v: dict,
    budget: int,
) -> dict:
    sources = (cycle, motion, consistency, i2v)
    scene_sets = [
        {key.rsplit("__c", 1)[0] for key in source if "__c" in key}
        for source in sources
    ]
    scenes = sorted(set.intersection(*scene_sets))
    if not scenes:
        raise RuntimeError("no shared complete scenes")

    picks: dict[str, int] = {}
    utilities: dict[str, list[float]] = {}
    for scene in scenes:
        cycle_values = candidate_values(cycle, scene, budget)
        motion_values = candidate_values(motion, scene, budget)
        consistency_values = candidate_values(consistency, scene, budget)
        i2v_values = candidate_values(i2v, scene, budget)

        global_evidence = (
            within_pool_zscore(-cycle_values)
            + within_pool_zscore(consistency_values)
            + within_pool_zscore(i2v_values)
        )
        selected = cva_select(global_evidence, motion_values)
        picks[scene] = int(selected.picks)
        utilities[scene] = selected.utility.tolist()

    return {
        "method": "CVA-Select",
        "formula": "z(G)+z(M)",
        "global_evidence": "z(-cycle)+z(consistency)+z(i2v)",
        "motion_evidence": "positive motion adequacy",
        "selection_uses_official_metrics": False,
        "candidate_budget": budget,
        "scenes": len(scenes),
        "picks": picks,
        "utilities": utilities,
    }


def selftest() -> None:
    budget = 4
    cycle = {f"scene__c{i:02d}": value for i, value in enumerate((3.0, 2.0, 1.0, 4.0))}
    motion = {f"scene__c{i:02d}": value for i, value in enumerate((0.0, 0.2, 0.9, 0.1))}
    consistency = {f"scene__c{i:02d}": value for i, value in enumerate((0.0, 0.2, 1.0, 0.1))}
    i2v = {f"scene__c{i:02d}": value for i, value in enumerate((0.0, 0.1, 0.8, 0.2))}
    result = freeze_picks(cycle, motion, consistency, i2v, budget)
    assert result["picks"] == {"scene": 2}
    print("PAI_CVA_SELECT_SELFTEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--consistency", type=Path)
    parser.add_argument("--i2v", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    required = (args.cycle, args.motion, args.consistency, args.i2v, args.output)
    if not all(required):
        parser.error("--cycle, --motion, --consistency, --i2v, and --output are required")
    if args.budget < 2:
        parser.error("--budget must be at least two")

    paths = {
        "cycle": args.cycle,
        "motion": args.motion,
        "consistency": args.consistency,
        "i2v": args.i2v,
    }
    payloads = {name: load_json(path) for name, path in paths.items()}
    result = freeze_picks(**payloads, budget=args.budget)
    result["input_sha256"] = {name: sha256(path) for name, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(f"PAI_CVA_SELECT_FROZEN scenes={result['scenes']} sha256={digest}")


if __name__ == "__main__":
    main()
