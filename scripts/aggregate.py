"""Collect test results of all experiments / seeds into one table, with paired tests against the baseline.

For every experiment: mean +- std across seeds of each metric's improvement (out - in), room accuracy,
speaker EER. Significance: per-utterance output metric averaged over seeds, Wilcoxon signed-rank test
against the baseline on the same test items.

  python scripts/aggregate.py --out_dir /content/drive/MyDrive/srse_runs --baseline B0
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["pesq", "estoi", "sisdr", "srmr"]


def fmt(mu, sd):
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        return "-"
    return f"{mu:.3f} ± {sd:.3f}" if sd == sd and sd > 0 else f"{mu:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--baseline", default="B0")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--groups", nargs="*", default=["sim_reverb", "sim_small", "sim_medium", "sim_large",
                                                     "sim_anechoic", "real_all"])
    a = ap.parse_args()
    root = Path(a.out_dir)

    summaries, utts = [], []
    for sj in sorted(root.glob(f"*/seed*/{a.tag}/summary.json")):
        s = json.loads(sj.read_text())
        exp, seed = sj.parents[2].name, sj.parents[1].name
        row = {"experiment": exp, "seed": seed, "room_acc": s.get("room_acc", math.nan),
               "speaker_eer": s.get("speaker_eer", math.nan), "rtf": s.get("rtf", math.nan)}
        for g, d in s["groups"].items():
            for mname in METRICS:
                if f"{mname}_delta" in d:
                    row[f"{g}:{mname}_delta"] = d[f"{mname}_delta"]
                    row[f"{g}:{mname}_out"] = d[f"{mname}_out"]
        summaries.append(row)
        df = pd.read_csv(sj.parent / "per_utt.csv")
        df["experiment"], df["seed"] = exp, seed
        utts.append(df)
    if not summaries:
        sys.exit(f"No {a.tag}/summary.json files under {root}")
    S = pd.DataFrame(summaries)
    U = pd.concat(utts, ignore_index=True)

    lines = [f"# Results ({a.tag})", "", f"Runs found: {len(S)} | baseline: {a.baseline}", ""]
    metrics = [mn for mn in METRICS if any(c.endswith(f":{mn}_delta") for c in S.columns)]
    for g in a.groups:
        cols = [f"{g}:{mn}_delta" for mn in metrics if f"{g}:{mn}_delta" in S.columns]
        if not cols:
            continue
        lines += [f"## {g} — improvement over unprocessed input (mean ± std over seeds)", ""]
        head = ["experiment", "seeds"] + [c.split(":")[1] for c in cols]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for exp, sub in S.groupby("experiment"):
            cells = [exp, str(len(sub))] + [fmt(sub[c].mean(), sub[c].std()) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Embedding quality (simulated test set, unseen speakers)", "",
              "| experiment | room acc | speaker EER | RTF |", "|---|---|---|---|"]
    for exp, sub in S.groupby("experiment"):
        lines.append(f"| {exp} | {fmt(sub.room_acc.mean(), sub.room_acc.std())} | "
                     f"{fmt(sub.speaker_eer.mean(), sub.speaker_eer.std())} | {sub.rtf.mean():.4f} |")
    lines.append("")

    if a.baseline in set(U.experiment):
        lines += [f"## Paired test vs {a.baseline} (per-utterance output metric averaged over seeds; Wilcoxon)", "",
                  "| experiment | set | metric | mean diff | p-value | n |", "|---|---|---|---|---|---|"]
        for mname in metrics:
            col = f"{mname}_out"
            if col not in U.columns:
                continue
            per = U.groupby(["experiment", "kind", "item"])[col].mean().reset_index()
            base = per[per.experiment == a.baseline].set_index(["kind", "item"])[col]
            for exp in sorted(set(per.experiment) - {a.baseline}):
                cur = per[per.experiment == exp].set_index(["kind", "item"])[col]
                for kind in ("sim", "real"):
                    try:
                        b_k, c_k = base.xs(kind), cur.xs(kind)
                    except KeyError:
                        continue
                    common = b_k.index.intersection(c_k.index)
                    d = (c_k.loc[common] - b_k.loc[common]).dropna()
                    if len(d) < 5 or np.allclose(d, 0):
                        continue
                    p = wilcoxon(d).pvalue
                    lines.append(f"| {exp} | {kind} | {mname} | {d.mean():+.4f} | {p:.2e} | {len(d)} |")
        lines.append("")

    probes = sorted(root.glob("*/seed*/probe.json"))
    if probes:
        lines += ["## Linear probes (speaker acc / room acc per tap)", "", "| run | P1 | P2 | P3 |", "|---|---|---|---|"]
        for pj in probes:
            p = json.loads(pj.read_text())["taps"]
            cells = [f"{p[t]['speaker_acc']:.3f} / {p[t]['room_acc']:.3f}" for t in ("P1", "P2", "P3")]
            lines.append(f"| {pj.parents[1].name}/{pj.parent.name} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines)
    (root / f"results_{a.tag}.md").write_text(text, encoding="utf-8")
    S.to_csv(root / f"results_{a.tag}_runs.csv", index=False)
    print(text)
    print(f"\n[saved] {root / f'results_{a.tag}.md'}")


if __name__ == "__main__":
    main()
