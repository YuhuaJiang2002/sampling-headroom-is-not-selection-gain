#!/usr/bin/env python3
"""Freeze OpenS2V CVA-Select picks, then evaluate all seven dimensions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cva_tts import cva_select


SOURCES = ("base", "qwen", "lingbot", "gpt54skill")
METRIC_FILES = {
    "aes_score": ("aesthetic_score.json",),
    "motion_amplitude": ("motion_amplitude.json",),
    "motion_smoothness": ("motion_smoothness.json",),
    "facesim_cur": ("facesim.json",),
    "gme_score": ("gmescore.json",),
    "nexus_score": ("nexusscore.json",),
    "natural_score": (
        "naturalscore_1.json",
        "naturalscore_2.json",
        "naturalscore_3.json",
    ),
}
RANGES = {
    "aes_score": (4.0, 7.0),
    "motion_amplitude": (0.0, 1.0),
    "motion_smoothness": (0.0, 1.0),
    "facesim_cur": (0.0, 1.0),
    "gme_score": (0.0, 1.0),
    "nexus_score": (0.0, 0.05),
    "natural_score": (1.0, 5.0),
}
WEIGHTS = {
    "aes_score": 0.16,
    "motion_smoothness": 0.06,
    "motion_amplitude": 0.02,
    "facesim_cur": 0.20,
    "gme_score": 0.12,
    "nexus_score": 0.20,
    "natural_score": 0.24,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_hashed_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def expected_ids(metric: str, all_ids: set[str]) -> set[str]:
    if metric == "facesim_cur":
        return {
            item
            for item in all_ids
            if item.startswith(("singlehuman_", "singleface_", "faceobj_", "humanobj_"))
        }
    if metric == "nexus_score":
        return {
            item
            for item in all_ids
            if not item.startswith(("singleface_", "multiface_"))
        }
    return all_ids


def load_metric(
    folder: Path, metric: str, all_ids: set[str]
) -> tuple[dict[str, float], dict[str, str]]:
    expected = expected_ids(metric, all_ids)
    payloads = []
    hashes = {}
    for filename in METRIC_FILES[metric]:
        path = folder / filename
        payload = load_json(path)
        if set(payload) != expected:
            raise RuntimeError(
                f"{path.name} coverage changed: expected={len(expected)} actual={len(payload)}"
            )
        payloads.append(payload)
        hashes[filename] = sha256(path)

    values = {}
    for item in expected:
        if metric == "aes_score":
            value = payloads[0][item]["aes_score"]
        elif metric == "motion_amplitude":
            value = payloads[0][item]["motion_fb"]
        elif metric == "motion_smoothness":
            value = payloads[0][item]["motion_smoothness"]
        elif metric == "facesim_cur":
            value = payloads[0][item]["cur_score"]
        elif metric == "gme_score":
            value = payloads[0][item]["gme_score"]
        elif metric == "nexus_score":
            value = payloads[0][item]["nexus_score"]
        else:
            value = float(np.mean([float(payload[item]) for payload in payloads]))
        values[item] = float(value)
    return values, hashes


def load_all_scores(
    folder: Path, all_ids: set[str]
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    rows = {item: {} for item in all_ids}
    hashes = {}
    for metric in METRIC_FILES:
        values, metric_hashes = load_metric(folder, metric, all_ids)
        for item, value in values.items():
            rows[item][metric] = value
        hashes.update({f"{metric}/{name}": digest for name, digest in metric_hashes.items()})
    return rows, hashes


def freeze_picks(
    graph: dict,
    motion_by_source: dict[str, dict[str, float]],
    all_ids: set[str],
) -> dict:
    if graph.get("frozen") is not True:
        raise RuntimeError("global evidence is not marked frozen")
    if graph.get("official_metrics_read_at_inference") is not False:
        raise RuntimeError("global evidence violates the outcome boundary")
    experiences = graph["graph"]["experience_subgraph"]
    picks = {}
    utilities = {}
    for item in sorted(all_ids):
        global_values = np.asarray([
            experiences[f"experience::{item}::{source}"]["deltas_vs_lingbot"][
                "component_mean"
            ]
            for source in SOURCES
        ])
        motion_values = np.asarray([
            motion_by_source[source][item] for source in SOURCES
        ])
        selected = cva_select(global_values, -motion_values)
        pick = int(selected.picks)
        picks[item] = SOURCES[pick]
        utilities[item] = selected.utility.tolist()
    return {
        "status": "OPENS2V_CVA_SELECT_PICKS_FROZEN",
        "method": "CVA-Select",
        "formula": "z(G)+z(M)",
        "global_evidence": "frozen same-task cross-seed component-mean utility",
        "motion_evidence": "negative frame-difference motion amplitude",
        "selection_uses_official_metrics": False,
        "candidate_sources": list(SOURCES),
        "items": len(all_ids),
        "picks": picks,
        "utilities": utilities,
    }


def normalize(metric: str, value: float) -> float:
    lower, upper = RANGES[metric]
    if metric in ("motion_amplitude", "motion_smoothness"):
        value = abs(value)
    clipped = min(max(value, lower), upper)
    return (clipped - lower) / (upper - lower)


def summarize(rows: dict[str, dict[str, float]]) -> dict:
    values = {metric: [] for metric in METRIC_FILES}
    for row in rows.values():
        for metric in values:
            if metric not in row:
                continue
            if metric == "nexus_score" and row[metric] == 0.0:
                continue
            values[metric].append(normalize(metric, row[metric]))
    if any(not metric_values for metric_values in values.values()):
        raise RuntimeError("selected rows do not cover all seven OpenS2V dimensions")
    means = {metric: float(np.mean(metric_values)) for metric, metric_values in values.items()}
    return {
        **means,
        "component_mean": float(np.mean(list(means.values()))),
        "weighted_score": float(sum(WEIGHTS[name] * value for name, value in means.items())),
        "counts": {name: len(metric_values) for name, metric_values in values.items()},
    }


def paired_category_bootstrap(deltas: dict[str, float], draws: int, seed: int) -> dict:
    categories: dict[str, list[str]] = {}
    for item in deltas:
        categories.setdefault(item.rsplit("_", 1)[0], []).append(item)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    ordered = sorted(categories)
    for draw in range(draws):
        selected = []
        for category in ordered:
            items = categories[category]
            indices = rng.integers(0, len(items), len(items))
            selected.extend(deltas[items[index]] for index in indices)
        samples[draw] = float(np.mean(selected))
    return {
        "delta": float(np.mean(list(deltas.values()))),
        "ci95": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "probability_positive": float(np.mean(samples > 0.0)),
        "draws": draws,
        "seed": seed,
    }


def run(args: argparse.Namespace) -> dict:
    graph = load_json(args.graph)
    metadata = load_json(args.metadata)
    all_ids = set(metadata)
    if len(all_ids) != args.expected_items:
        raise RuntimeError(f"expected {args.expected_items} items, found {len(all_ids)}")
    source_folders = dict(entry.split("=", 1) for entry in args.score)
    if set(source_folders) != set(SOURCES):
        raise RuntimeError(f"--score must cover exactly {SOURCES}")

    motion_by_source = {}
    selection_hashes = {
        "graph": sha256(args.graph),
        "metadata": sha256(args.metadata),
    }
    for source in SOURCES:
        values, hashes = load_metric(
            Path(source_folders[source]), "motion_amplitude", all_ids
        )
        motion_by_source[source] = values
        selection_hashes[f"{source}/motion_amplitude.json"] = hashes[
            "motion_amplitude.json"
        ]
    frozen = freeze_picks(graph, motion_by_source, all_ids)
    frozen["selection_input_sha256"] = selection_hashes
    picks_sha256 = write_hashed_json(args.picks_output, frozen)

    source_rows = {}
    analysis_hashes = {}
    for source in SOURCES:
        source_rows[source], hashes = load_all_scores(
            Path(source_folders[source]), all_ids
        )
        analysis_hashes.update({f"{source}/{name}": digest for name, digest in hashes.items()})
    selected_rows = {
        item: source_rows[source][item] for item, source in frozen["picks"].items()
    }
    source_summaries = {
        source: summarize(rows) for source, rows in source_rows.items()
    }
    cva_summary = summarize(selected_rows)
    best_source = max(
        SOURCES, key=lambda source: source_summaries[source]["component_mean"]
    )
    per_item_delta = {}
    for item in sorted(all_ids):
        selected_values = [
            normalize(metric, value)
            for metric, value in selected_rows[item].items()
            if not (metric == "nexus_score" and value == 0.0)
        ]
        baseline_values = [
            normalize(metric, value)
            for metric, value in source_rows[best_source][item].items()
            if not (metric == "nexus_score" and value == 0.0)
        ]
        per_item_delta[item] = float(np.mean(selected_values) - np.mean(baseline_values))

    result = {
        "status": "OPENS2V_CVA_SELECT_ANALYSIS_COMPLETE",
        "evidence_status": "post_hoc_descriptive_only",
        "method": "CVA-Select",
        "formula": "z(G)+z(M)",
        "selection_uses_official_metrics": False,
        "population": {
            "benchmark": "OpenS2V-Eval v1.1 Open-Domain matched pool",
            "items": len(all_ids),
            "candidate_sources": list(SOURCES),
            "leaderboard_comparable": False,
        },
        "source_summaries": source_summaries,
        "cva_select_summary": cva_summary,
        "best_single_source": best_source,
        "cva_select_minus_best_single_component_mean": paired_category_bootstrap(
            per_item_delta, args.bootstrap_draws, args.seed
        ),
        "pick_counts": {
            source: sum(value == source for value in frozen["picks"].values())
            for source in SOURCES
        },
        "inputs": {
            "frozen_picks_sha256": picks_sha256,
            "analysis_input_sha256": analysis_hashes,
        },
    }
    digest = write_hashed_json(args.output, result)
    print(
        "OPENS2V_CVA_SELECT_COMPLETE "
        f"mean={cva_summary['component_mean']:.9f} sha256={digest}"
    )
    return result


def selftest() -> None:
    selection = cva_select(
        np.asarray([0.0, 0.2, 1.0, 0.1]),
        np.asarray([0.0, 0.1, 0.9, 0.2]),
    )
    assert int(selection.picks) == 2
    assert normalize("aes_score", 5.5) == 0.5
    assert normalize("natural_score", 3.0) == 0.5
    print("OPENS2V_CVA_SELECT_SELFTEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "selftest"))
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--score", action="append", default=[])
    parser.add_argument("--picks-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-items", type=int, default=180)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        return
    required = (args.graph, args.metadata, args.picks_output, args.output)
    if not all(required) or not args.score:
        parser.error(
            "run requires --graph, --metadata, four --score source=folder entries, "
            "--picks-output, and --output"
        )
    run(args)


if __name__ == "__main__":
    main()
