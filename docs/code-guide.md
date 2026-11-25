# Code Guide

How to run the single-speaker study described in [research-plan.md](research-plan.md): does speaker and/or room-acoustic information help speech enhancement (SE), ASR, both, or neither?

## Quick start on Google Colab

Open [notebooks/colab_quickstart.ipynb](../notebooks/colab_quickstart.ipynb) in Colab (File → Open notebook → GitHub), select a **T4 GPU** runtime, and run the cells top to bottom. Results and checkpoints are written to Google Drive (`MyDrive/srse_runs`), so a disconnect loses nothing: re-running a cell skips finished runs and resumes unfinished ones from `last.pt`.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Data | `scripts/prepare_data.py` | LibriSpeech train-clean-100 + test-clean, RIRS_NOISES; JSONL manifests |
| 2. Train + validate | `scripts/train.py` | `<out_dir>/<experiment>/seed<k>/` with `best.pt`, `last.pt`, `metrics.jsonl`, `config.json` |
| 3. Test | `scripts/test.py` | `<run>/test/per_utt.csv`, `summary.json` |
| 4. Probes | `scripts/probe.py` | `<run>/probe.json` (speaker / room linear-probe accuracy at P1, P2, P3) |
| 5. Results | `scripts/aggregate.py` | `<out_dir>/results_test.md` — mean ± std over seeds and paired Wilcoxon tests against B0 |
| Grid | `scripts/run_grid.py` | Runs 2–4 for many experiments × seeds; skips finished runs |
| Export | `scripts/export_reverb.py` | Reverberation-augmented WAV files + manifest (for listening or ASR) |
| Check | `scripts/check_causality.py` | Verifies the model is causal; reports parameters, latency and real-time factor |

## Data recipe (follows EUSIPCO 2022)

- **Speech:** LibriSpeech train-clean-100. The small setting uses all 251 speakers × 40 utterances (~35 h); set `data.utts_per_speaker: null` for all ~100 h. Two utterances per speaker are held out for validation (closed-set speaker accuracy).
- **Rooms:** each training example is anechoic, small, medium or large with equal probability. Simulated RIRs come from RIRS_NOISES (Ko et al. 2017), already labelled by room size. RIRs are split **by room**: 80% train, 10% validation, 10% test, so test rooms are never seen in training.
- **Target:** direct path + first 50 ms of reflections (`data.early_ms`). For anechoic items, input = target, which tests that the model does no harm.
- **Test sets** (unseen speakers from test-clean, fixed by `data.test_seed`, so identical for every experiment):
  - `sim`: test-split simulated RIRs, 200 utterances per room class.
  - `real`: 300 items with measured RIRs from RIRS_NOISES, binned by estimated T60 (< 0.3 s, 0.3–0.6 s, > 0.6 s), because real RIRs have no room-size label.
- Data is generated on the fly, so every epoch sees new crops, rooms and levels.

## Exporting the augmented audio

Training never writes reverberant audio to disk. To listen to it, or to build ASR training and test data, `scripts/export_reverb.py` renders items with exactly the same code as training. The exported files match what the model sees, up to 16-bit rounding.

Each item gives `<id>_reverb.wav` (input), `<id>_target.wav` (direct path + 50 ms) and, with `--dry`, `<id>_dry.wav`. A `manifest.jsonl` / `manifest.csv` records speaker, room class, RIR, T60 (real RIRs), level and, for full utterances, the LibriSpeech transcript.

| Goal | Command |
| --- | --- |
| Listen to 40 training examples (10 per room class) | `python scripts/export_reverb.py --split train --n 40 --dry --out_dir /content/reverb_examples` |
| Exactly the items the test script scores | `python scripts/export_reverb.py --split test --match_eval --out_dir /content/reverb_test` (also `val`, `real`) |
| Only one room class | add `--room large` |
| ASR training data: every training utterance once, full length, random room, with transcripts | `python scripts/export_reverb.py --split train --all --full --out_dir /content/reverb_asr_train` |

Splits: `train` (training speakers, training rooms), `val` (held-out utterances, validation rooms), `test` (test-clean speakers, unseen simulated rooms), `real` (test-clean speakers, measured RIRs).

Disk space: the small setting's training utterances at full length are roughly 35 h, about 4 GB per stream (reverb, target, dry) as 16-bit WAV; all of train-clean-100 is about 11 GB per stream. Write to `/content` and copy a zip to Drive rather than writing thousands of files directly to Drive.

## Model

Causal convolutional-recurrent mask network (`srse/models/crn.py`), ~2.3 M parameters:

```
reverberant STFT (20 ms window, 10 ms hop) -> compressed complex features
  -> 5 causal conv layers ............ P1
  -> 2-layer unidirectional GRU ...... P2   (default tap)
  -> 5 deconv layers with skips ...... P3 = penultimate layer
  -> complex mask (|M| < 1) x input -> iSTFT
```

**MTL heads** (`srse/models/heads.py`), attached at the tap set by `mtl.tap`:

```
tap -> 2 x Linear(256) -> causal stats pooling (mean + std over the last N frames)
    -> Linear(256) = MTL EMBEDDING -> speaker head (2 x 256 -> speakers)
                                   -> room head    (2 x 256 -> 4 classes)
```

The pooling is causal, so the embedding exists at every frame and can be used in streaming. During training N is drawn per batch from `mtl.pool_frames` (0.25 s to the whole segment), so the heads learn to work with short context.

## How λ works

Loss: `L = L_SE + L_spk_head + L_room_head`. The heads always learn. What `λ` controls is how much of each head's gradient reaches the enhancement backbone, via a gradient-scaling layer at the tap:

| `lambda_*` | Effect on the backbone | Used in |
| --- | --- | --- |
| > 0 | Multi-task learning: backbone receives λ × task gradient (same as `L_SE + λ·L_task`) | S, R, M runs |
| 0 | Nothing; heads only observe (an online probe) | B0 |
| < 0 | Gradient reversal: backbone is pushed to **remove** that information | C_adv runs |

So every run, including the baseline, reports how much speaker and room information is readable at the tap.

## Experiments

| Config | λ_spk | λ_room | Purpose |
| --- | --- | --- | --- |
| `B0` | 0 | 0 | Baseline |
| `S1`, `S2` | 0.1, 1 | 0 | Speaker only (H1) |
| `R1`, `R2` | 0 | 0.1, 1 | Room only (H2) |
| `M1`, `M2`, `M3` | (1, 1), (1, 0.1), (0.1, 1) | | Both — the EUSIPCO 2022 grid |
| `C_shuf` | 1 | 1 | Same as M1 with random labels — does the *information* help, or just an extra loss? |
| `C_adv_spk`, `C_adv_room` | −0.1 / 0 | 0 / −0.1 | Remove the information — does SE get worse without it? |
| `O_room` | 0 | 0 | True room class given to the network (FiLM) — upper bound |

Variants without new files, for example:

```bash
python scripts/run_grid.py --experiments M1 --seeds 0 --set experiment=M1_P3 mtl.tap=P3      # head position
python scripts/run_grid.py --experiments M1 --seeds 0 --set experiment=M1_fb mtl.feedback=true  # feed embedding back
python scripts/test.py --run_dir <runs>/M1/seed0 --window 100 --tag test_w100                  # 1 s context
```

## Reading the outputs

- `metrics.jsonl` (per epoch): train losses, `val_se_loss` (used for the best checkpoint and early stopping), `val_sisdri` (reverberant items), `val_spk_acc`, `val_room_acc`.
- `test/summary.json`: per group (`sim_reverb`, `sim_<class>`, `real_all`, `real_t60_*`) the input and output PESQ, ESTOI and SI-SDR and their improvement. Also room accuracy, room accuracy vs seconds since utterance start (`room_acc_vs_time`), speaker EER on unseen speakers, and the real-time factor.
- `results_test.md`: one table per group across experiments, embedding quality, paired tests against B0, and probe results.

The suggested decision rule (research plan §11): an effect counts if it is significant across seeds **and** at least +0.05 PESQ or +0.3 dB SI-SDR.

## Running locally without the full corpora

```bash
python tests/make_synthetic_data.py --data_root /tmp/fake
python scripts/prepare_data.py --data_root /tmp/fake --skip_download
python scripts/run_grid.py --experiments B0 M1 --seeds 0 --set paths.data_root=/tmp/fake \
    paths.manifest_dir=/tmp/fake/manifests paths.out_dir=/tmp/runs train.epochs=1 train.batch_size=4 \
    data.num_workers=0 data.val_items_per_class=4 data.test_utts=3 data.real_rir_test_utts=4
```

The synthetic audio is not speech; this only checks that the code runs.

## Not yet included

- Streaming ASR adaptation (plan §10) — next stage; the embedding is already returned per frame (`out["emb"]`).
- DNSMOS, and SRMR (optional: `pip install git+https://github.com/jfsantos/SRMRpy`, then `test.py --srmr`).
- Frame-by-frame streaming inference with explicit state. The model is verified causal and runs well under real time, so offline full-utterance inference gives identical outputs.
