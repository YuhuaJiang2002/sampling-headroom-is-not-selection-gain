# Sampling Headroom Is Not Selection Gain

Implementation and experiment code for **“Sampling Headroom Is Not
Selection Gain: A Compute-Value Audit of Test-Time Scaling for Video World
Models.”**

This repository is intentionally narrow. It contains only the paper's method
and the experiment entry points needed to apply it. Manuscript source, PDFs,
figures, tables, result dumps, model weights, datasets, cloud configuration,
and internal development code are not included.

## Contents

```text
src/cva_tts/
  audit.py          Compute-Value Audit quantities and four-stage decision
  selection.py      CVA-Select and its full G/N/M component factorial

experiments/
  physics_iq_ablation.py   Physics-IQ seven-configuration ablation
  pai_cva_select.py        PAI-Bench-robot CVA-Select pick freezer
  opens2v_cva_select.py    OpenS2V CVA-Select and seven-axis analysis
```

## Method

The Compute-Value Audit (CVA) distinguishes four requirements for useful
test-time scaling: additional sampling must create **opportunity**, observable
signals must provide a predictive **state**, that state must support a
beneficial **action**, and the gain must exceed the full compute **fee**.

The common selector used by the released experiments is

```text
CVA-Select(z) = zscore(G(z)) + zscore(M(z)),
```

where both terms are standardized within the candidate pool and oriented so
that larger values are better. `G` is benchmark-specific global/non-motion
evidence and `M` is benchmark-specific motion evidence. Physics-IQ additionally
uses `N` only for the seven-way component ablation; `N` is not part of the
selected `G+M` rule.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Smoke checks

Each experiment has a deterministic, data-free self-test:

```bash
python experiments/physics_iq_ablation.py --selftest
python experiments/pai_cva_select.py --selftest
python experiments/opens2v_cva_select.py selftest
```

## Running the experiments

The scripts expose their complete data contracts through `--help`. The
Physics-IQ entry point takes separate signal and outcome files so it can write
and hash every pick before opening official quality. The OpenS2V entry point
likewise freezes picks from `G` and motion evidence before loading the seven
evaluation dimensions. PAI writes only frozen picks and never consumes the
official metric.

```bash
python experiments/physics_iq_ablation.py --help
python experiments/pai_cva_select.py --help
python experiments/opens2v_cva_select.py --help
```

Candidate videos, benchmark annotations, and pretrained checkpoints must be
obtained from their original providers under the corresponding licenses.
