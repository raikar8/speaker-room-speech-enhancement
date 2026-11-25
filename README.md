# Speaker Room Speech Enhancement

Does knowing **who is speaking** and **what room they are in** help a real-time speech enhancement network, a streaming ASR system, both, or neither?

A causal mask-based enhancement network is trained with auxiliary speaker and room-size (anechoic / small / medium / large) classification heads on an internal layer. Controlled experiments vary how much of that information reaches the network. The learned embedding is later reused to adapt streaming ASR. Data preparation follows Raikar et al., EUSIPCO 2022.

## Documents

- [Literature survey](docs/literature-survey.md): problem definition, architectures, clue types, key papers (2017–2026), training and real-time inference, datasets, metrics, open challenges, and a suggested project plan.
- [Research plan](docs/research-plan.md): single-speaker study of whether speaker and room-acoustic information help speech enhancement, streaming ASR, both, or neither — hypotheses, architecture, controls, experiment grid and decision criteria.
- [Code guide](docs/code-guide.md): how to run data preparation, training, testing, probes and result aggregation.

## Quick start (Google Colab)

Open [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb) in Colab with a T4 GPU and run the cells in order.

Or from a terminal with a GPU:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --data_root /content/data
python scripts/check_causality.py --config configs/experiments/M1.yaml
python scripts/run_grid.py --experiments B0 S2 R2 M1 --seeds 0 --probe
python scripts/aggregate.py --out_dir /content/drive/MyDrive/srse_runs --baseline B0
```

Paths default to Colab locations (`configs/base.yaml`); override them with `--set paths.data_root=... paths.out_dir=...`.

## Layout

```
configs/            base.yaml + one file per experiment (B0, S1–S2, R1–R2, M1–M3, controls, oracle)
srse/               package: data (download, manifests, on-the-fly reverberation), model (causal CRN + MTL heads), losses, metrics
scripts/            prepare_data, train, test, probe, aggregate, run_grid, check_causality
notebooks/          Colab quickstart
tests/              synthetic-data generator for a fast end-to-end check
docs/               literature survey, research plan, code guide
```
