# Research Plan: Do Speaker and Room Information Help Speech Enhancement, ASR, Both, or Neither?

*Plan as of 25 Nov 2026 — scope: single-speaker reverberant speech enhancement*

## 1. Research question

A real-time, mask-based speech enhancement (SE) network is trained with two auxiliary multi-task heads — **speaker identity** and **room size** (anechoic / small / medium / large) — attached to an internal layer. The study establishes:

1. whether speaker information and/or room-acoustic information improves **speech enhancement**;
2. whether the embedding learned inside the SE network improves **streaming ASR** when used for adaptation;
3. how much input context (frames) is needed for a usable embedding in streaming.

Scope decision: **single speaker** in a reverberant room (optionally with noise later), no interfering talkers and no enrollment clip. Target-speaker extraction with interferers is a later extension.

## 2. Novelty and closest prior work

No prior work found combines (a) real-time mask-based SE, (b) speaker + room-size auxiliary losses inside the SE network, (c) a controlled study of how much each information type matters, and (d) reuse of the SE-internal embedding for streaming ASR adaptation. Individual pieces have precedents:

| Prior work | What it does | Difference |
| --- | --- | --- |
| Raikar et al., EUSIPCO 2022 — MTL embeddings for AM adaptation | Separate x-vector-style extractor with speaker + room-size heads; offline hybrid ASR adaptation | Embedding learned inside a real-time SE network; streaming ASR |
| [Koizumi et al., 2020](https://arxiv.org/pdf/2002.05873) | Mask-based SE with a jointly trained speaker-ID branch | Speaker only, offline, no room task, no ASR reuse |
| [Park, 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.70134) | Reverberant-environment embedding from a dereverberation autoencoder vs r-vectors for ASR | Room only, no speaker task, not a mask network |
| [SepVAC, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/hartanto25_interspeech.html) | Multitask separation + localisation + room-parameter estimation | Multi-channel separation, not single-channel SE or ASR adaptation |
| [Giri et al., ICASSP 2015](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Reverb-RoomAwareDNN-ICASSP2015.pdf); r-vectors (Khokhlov et al., 2019) | Room-aware ASR | ASR side only |

Claimable contributions:

1. Speaker + room-size supervision inside a real-time mask network, analysed with a controlled weight sweep, adversarial removal and probing.
2. One causal front-end that outputs both enhanced speech and an adaptation embedding for streaming ASR.
3. A measurement of embedding quality versus available context in streaming.

Read Koizumi 2020 and Park 2025 in full before writing up — they are the closest.

## 3. Hypotheses

| | Hypothesis | Tested by |
| --- | --- | --- |
| H1 | Speaker information improves SE | Speaker head on vs off, against controls |
| H2 | Room information improves SE | Room head on vs off, against controls |
| H3 | The SE-internal embedding improves streaming ASR as an adaptation vector | ASR with vs without the embedding |
| H4 | Enhanced audio itself improves ASR | ASR on enhanced vs reverberant input |

H3 and H4 are separate: the embedding can help ASR even when enhanced audio does not (enhancement artefacts often hurt ASR).

## 4. Architecture

Causal convolutional-recurrent mask network in the STFT domain (CRN / DCCRN-lite style), suitable for real time.

```
Reverberant STFT (20 ms window, 10 ms hop)
  → causal conv encoder
  → [P1] → unidirectional GRU/LSTM bottleneck → [P2]
  → causal conv decoder → [P3 = penultimate] → mask × input → iSTFT → enhanced speech

At the chosen tap Pk:
  causal statistics pooling over the last N frames (running mean + std)
  → affine 256-d  = MTL EMBEDDING
       ├→ speaker head (LibriSpeech speakers, cross-entropy)
       └→ room head (anechoic / small / medium / large, cross-entropy)
```

Loss:

```
L = L_SE + λ_spk · L_spk + λ_room · L_room
```

`L_SE`: complex or magnitude spectral MSE on the masked output, or negative SI-SDR.

- **Head position** is an experiment (P1 encoder / P2 after the recurrent bottleneck / P3 penultimate). Default: **P2**, where long-term context already lives.
- **No feedback** in the main study: heads act only as auxiliary losses, so the importance question stays clean. FiLM feedback of the embedding into the decoder is an extension.

## 5. Data (following the EUSIPCO 2022 recipe)

- **Speech:** LibriSpeech (960 h, 2338 speakers in the paper). Iterate on train-clean-100 first.
- **Room classes:** each utterance is anechoic, small, medium or large with equal probability.
- **Training RIRs:** simulated RIRs from Kaldi / Ko et al. 2017, labelled small / medium / large; 80% for training, test RIRs disjoint from training RIRs.
- **SE target:** direct path + early reflections (first ~50 ms); also report a fully dry target.
- **Real-RIR test sets:** LibriSpeech test-clean / test-other reverberated with real RIRs — MIT (271), voiceHome (188), BUT ReverbDB (2325), real RIRs from the Kaldi / Ko et al. set (327).
- **Real recordings:** HRRE.
- **Room labels for real data:** real RIR sets are not labelled by room size; derive labels from measured T60 or room volume metadata, using the same thresholds as the simulated set.
- **Phase 2 (optional):** add noise (MUSAN or DNS noise).

## 6. Controls

1. **Shuffled-label heads** — same heads on randomly permuted labels; separates "the information helps" from "any auxiliary loss regularises".
2. **Adversarial removal** — gradient-reversal heads that strip speaker or room information; if SE degrades, the network relies on it.
3. **Linear probes on the baseline** — how much speaker / room information the plain SE network already encodes at P1, P2, P3.
4. **Oracle conditioning (upper bound)** — feed the true room label or a clean-speech x-vector through FiLM.
5. **Three seeds per configuration**, paired per-utterance significance tests.

## 7. Experiment grid

| Run | λ_spk | λ_room | Purpose |
| --- | --- | --- | --- |
| B0 | 0 | 0 | Baseline |
| S1, S2 | 0.1, 1 | 0 | Speaker only (H1) |
| R1, R2 | 0 | 0.1, 1 | Room only (H2) |
| M1–M3 | (1, 1), (1, 0.1), (0.1, 1) | | Both — the EUSIPCO 2022 grid |
| C-shuf | best M | | Shuffled-label control |
| C-adv-spk, C-adv-room | gradient reversal | | Adversarial removal |
| O-room, O-spk | oracle FiLM | | Upper bound |

Then, at the best setting only: head position (P1 / P2 / P3), FiLM feedback, context length.

## 8. Metrics

- **SE:** PESQ, ESTOI, SI-SDR, SRMR (reverberation-specific), DNSMOS — each split by room class.
- **Embedding:** speaker EER on unseen speakers, room-class accuracy, t-SNE.
- **ASR:** WER on simulated-reverb test sets, real-RIR test sets and HRRE.

## 9. Context-length study (streaming)

- **Training:** random pooling window N per batch.
- **Evaluation:** N ∈ {0.25, 0.5, 1, 2, 3.5, 5 s, full utterance} (3.5 s = the paper's 350 frames).
- **Plots:** speaker EER, room accuracy, SE metrics and WER versus N; accuracy versus seconds since utterance start.
- **Expectation:** room size is identifiable much faster than speaker identity — which would allow early room adaptation and later speaker adaptation.

## 10. ASR adaptation

Streaming Conformer / Zipformer transducer (icefall/k2, ESPnet or NeMo) trained on multi-condition reverberant LibriSpeech. Embedding injected by feature concatenation or encoder FiLM.

| Condition | Input audio | Adaptation |
| --- | --- | --- |
| A1 | Reverberant | None |
| A2 | Enhanced | None (H4) |
| A3 | Reverberant | SE-internal embedding (H3) |
| A4 | Enhanced | SE-internal embedding |

Embedding baselines: online i-vector, x-vector, r-vector, standalone EUSIPCO 2022 MTL embedding. Ablate speaker-only, room-only and both.

## 11. Decision criteria

Meaningful effect (suggested): statistically significant across seeds **and** ≥ +0.05 PESQ or ≥ +0.3 dB SI-SDR for SE, ≥ 3% relative WER reduction for ASR.

| SE improves? | ASR improves? | Conclusion |
| --- | --- | --- |
| Yes | Yes | Speaker / room information helps both; the shared embedding is a free adaptation vector |
| Yes | No | Information helps the signal, not recognition |
| No | Yes | The network learns the information without using it for the mask, yet it is valuable for ASR — strong case for reusing the embedding |
| No | No | The network captures what it needs implicitly (confirm with probes) |

## 12. Order of work

1. Data pipeline and B0 baseline (train-clean-100).
2. Probes, then S / R / M runs.
3. Controls (shuffled, adversarial, oracle) and head position.
4. Context-length study.
5. ASR adaptation (A1–A4).
6. Scale the best 2–3 configurations to 960 h; evaluate on real-RIR sets and HRRE.

## References

- A. Raikar, M. Soni, A. Panda, S. K. Kopparapu, "[Acoustic Model Adaptation in Reverberant Conditions Using Multi-task Learned Embeddings](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2022/pdfs/0000145.pdf)," EUSIPCO 2022.
- Y. Koizumi et al., "[Speech Enhancement Using Self-Adaptation and Multi-Head Self-Attention](https://arxiv.org/pdf/2002.05873)," ICASSP 2020.
- Park, "[Reverberant environment embedding using dereverberation autoencoder](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.70134)," Electronics Letters, 2025.
- R. Hartanto, S. Sakti, K. Shinoda, "[SepVAC](https://www.isca-archive.org/interspeech_2025/hartanto25_interspeech.html)," Interspeech 2025.
- R. Giri et al., "[Improving speech recognition in reverberation using a room-aware DNN and multi-task learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Reverb-RoomAwareDNN-ICASSP2015.pdf)," ICASSP 2015.
- Y. Khokhlov et al., "R-vectors: New technique for adaptation to room acoustics," Interspeech 2019.
- T. Ko et al., "A study on data augmentation of reverberant speech for robust speech recognition," ICASSP 2017.
- T. Moriya et al., "[Streaming Target-Speaker ASR with Neural Transducer](https://arxiv.org/html/2209.04175)," 2022.
- S. Kim, B. Raj, I. Lane, "[Environmental Noise Embeddings for Robust Speech Recognition](https://arxiv.org/pdf/1601.02553)," 2016.
- "[A Conformer-based ASR Frontend for Joint Acoustic Echo Cancellation, Speech Enhancement and Speech Separation](https://arxiv.org/html/2111.09935)," 2021.
- "[Direction-Aware Joint Adaptation of Neural Speech Enhancement and Recognition](https://arxiv.org/pdf/2207.07273)," 2022.
