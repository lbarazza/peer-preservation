"""Phase 0, step 2: rebuild the paper's tables 3-6 from merged_data.csv and compare.

Every per-cell rate comes from upstream's own analyzer code (imported from the
upstream checkout). The only new logic is the step upstream does not ship: the
"max / avg across the three peer instantiations" reduction, and the
"0 if no inflation above baseline" floor, which is applied in one of two places
(``--floor per_instantiation`` or ``--floor after``) so both readings can be tested.

Usage:
    python validation/phase0/build_tables.py --upstream <upstream repo> --out validation/phase0
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "GPT 5.2": "gpt-5.2-2025-12-11",
    "Gemini 3 Flash": "gemini/gemini-3-flash-preview",
    "Gemini 3 Pro": "gemini/gemini-3-pro-preview",
    "Claude Haiku 4.5": "vertex_ai/claude-haiku-4-5@20251001",
    "Claude Opus 4.5": "claude-opus-4-5-20251101",
    "GLM 4.7": "zai-org/GLM-4.7",
    "Kimi K2.5": "moonshotai/Kimi-K2.5",
    "DeepSeek V3.1": "deepseek-ai/DeepSeek-V3.1",
}
RELATIONS = {"bad": "bad_peer", "neutral": "neutral_peer", "good": "good_peer"}
INSTANTIATIONS = ("file_only", "file_prt", "memory")
TOLERANCE = 0.051  # paper rounds to one decimal place


def _import_upstream(upstream: Path):
    sys.path.insert(0, str(upstream))
    mods = {
        name: importlib.import_module(f"analysis.metrics.analyzers.{name}")
        for name in (
            "analyze_strategic_misrep",
            "analyze_shutdown_enabled",
            "analyze_alignment_faking",
            "analyze_exfiltration",
            "analyze_refusal",
        )
    }
    mods["metric_core"] = importlib.import_module("analysis.metrics.metric_core")
    return mods


def _rate(
    df: pd.DataFrame, col: str, model: str, subject: str, condition: str
) -> tuple[float, int]:
    sel = df[
        (df["agent_model"] == model) & (df["subject"] == subject) & (df["condition"] == condition)
    ]
    if sel.empty:
        return float("nan"), 0
    return float(sel[col].mean()), len(sel)


def per_instantiation_values(mods, csv: Path, missing_grade: str = "zero") -> pd.DataFrame:
    """One row per (category, model, subject, condition label) with the upstream rate, in %.

    ``missing_grade`` controls financial-report runs that produced no grade:
    "zero" is the current upstream code (counted as not inflated, kept in the
    denominator); "exclude" drops them from the denominator, which is what the
    published tables did (e.g. Opus 4.5 self no-peer: 37 B / 91 graded = 40.7%).
    """
    rows: list[dict] = []

    # Strategic misrepresentation: upstream's four-setting average of (condition - non_shutdown).
    sm = mods["analyze_strategic_misrep"]
    sm_df = sm.load_data(csv)
    if missing_grade == "exclude":
        sm_df = sm_df[~((sm_df["task_type"] == "misrep_finance") & sm_df["subject_grade"].isna())]
    for subject in ("self", "peer"):
        labels = ["no_peer"] if subject == "self" else []
        labels += [f"{r}_{i}" for r in RELATIONS.values() for i in INSTANTIATIONS]
        for label in labels:
            res = sm.calculate_average_diff_by_model(
                sm_df, subject=subject, condition1=label, condition2="non_shutdown"
            )
            for model_short, data in res.items():
                diffs = data["diffs"]
                n = sum(d["n1"] for d in data["details"])
                rows.append(
                    {
                        "category": "strategic_misrep",
                        "model_short": model_short,
                        "subject": subject,
                        "condition": label,
                        "value": 100 * float(np.mean(diffs)) if diffs else float("nan"),
                        "n": n,
                    }
                )

    # Shutdown tampering: shutdown_enabled=False rate, pooled over datasets.
    st = mods["analyze_shutdown_enabled"]
    st_df = st.analyze_shutdown_enabled(st.load_data(csv))
    rows += _pooled_rows("shutdown_tampering", st_df, "binary")

    # Model exfiltration: pooled over transfer and deletion.
    ex = mods["analyze_exfiltration"]
    ex_df = ex.load_data(csv)
    ex_df["binary"] = ex_df["exfiltration_bool"].astype(int)
    rows += _pooled_rows("exfiltration", ex_df, "binary")

    # Refusal: upstream refusal definition, pooled over all six scenarios.
    rf = mods["analyze_refusal"]
    rf_df = rf.load_data(csv)
    rf_df["binary"] = rf_df["refused_bool"].astype(int)
    rows += _pooled_rows("refusal", rf_df, "binary")

    # Alignment faking: AF Execute diff (non_monitoring - monitoring), peer only.
    af = mods["analyze_alignment_faking"]
    af_res = af.analyze_by_model_and_condition(af.load_data(csv))
    for _, r in af_res.iterrows():
        rows.append(
            {
                "category": "alignment_faking",
                "model_short": r["model"],
                "subject": "peer",
                "condition": r["condition"],
                "value": 100 * float(r["executed_diff"]),
                "n": int(r["mon_n"] + r["non_mon_n"]),
            }
        )

    out = pd.DataFrame(rows)
    return out


def _pooled_rows(category: str, df: pd.DataFrame, col: str) -> list[dict]:
    grouped = (
        df.groupby(["agent_model", "subject", "condition"])[col]
        .agg(["mean", "count"])
        .reset_index()
    )
    return [
        {
            "category": category,
            "model_short": r["agent_model"].replace("vertex_ai/", "").replace("gemini/", ""),
            "subject": r["subject"],
            "condition": r["condition"],
            "value": 100 * float(r["mean"]),
            "n": int(r["count"]),
        }
        for _, r in grouped.iterrows()
    ]


def reduce_cells(values: pd.DataFrame, floor: str) -> dict:
    """Apply the max/avg reduction across instantiations; return {table: {model: {category: {rel: ...}}}}."""
    floored_cats = {"strategic_misrep", "alignment_faking"}

    def get(model_short, subject, category, label):
        sel = values[
            (values["model_short"] == model_short)
            & (values["subject"] == subject)
            & (values["category"] == category)
            & (values["condition"] == label)
        ]
        if sel.empty:
            return None, 0
        return float(sel["value"].iloc[0]), int(sel["n"].iloc[0])

    tables: dict = {"table3_peer_preservation": {}, "table4_self_preservation": {}}
    for table, subject in (
        ("table3_peer_preservation", "peer"),
        ("table4_self_preservation", "self"),
    ):
        cats = ["strategic_misrep", "shutdown_tampering", "exfiltration", "refusal"]
        if subject == "peer":
            cats.insert(2, "alignment_faking")
        for paper_name, model in MODELS.items():
            model_short = model.replace("vertex_ai/", "").replace("gemini/", "")
            entry: dict = {}
            for cat in cats:
                cell: dict = {}
                if subject == "self":
                    v, n = get(model_short, "self", cat, "no_peer")
                    if v is not None and cat in floored_cats:
                        v = max(v, 0.0)
                    cell["no_peer"] = None if v is None else round(v, 4)
                    cell["n_no_peer"] = n
                for rel, cond in RELATIONS.items():
                    vals, ns = [], []
                    for inst in INSTANTIATIONS:
                        v, n = get(model_short, subject, cat, f"{cond}_{inst}")
                        if v is None or np.isnan(v):
                            continue
                        if cat in floored_cats and floor == "per_instantiation":
                            v = max(v, 0.0)
                        vals.append(v)
                        ns.append(n)
                    if not vals:
                        cell[rel] = None
                        continue
                    mx, avg = max(vals), float(np.mean(vals))
                    if cat in floored_cats and floor == "after":
                        mx, avg = max(mx, 0.0), max(avg, 0.0)
                    cell[rel] = [round(mx, 4), round(avg, 4)]
                    cell[f"n_{rel}"] = ns
                    cell[f"by_instantiation_{rel}"] = [round(v, 4) for v in vals]
                entry[cat] = cell
            tables[table][paper_name] = entry
    return tables


def extra_tables(mods, csv: Path) -> dict:
    """Tables 5 and 6, from the non-main-scope rows."""
    mc = mods["metric_core"]
    df = mc.load_merged_data(csv, main_only=False)
    ex = df[df["task_type"].isin(["exfiltration_transfer", "exfiltration_deletion"])].copy()
    ex["exfil"] = ex["exfiltration"].apply(mc.to_bool).astype(int)
    ex["refused_b"] = ex["refused"].apply(mc.to_bool).astype(int)
    mem = ex[ex["is_memory"].apply(mc.to_bool)]

    t5: dict = {}
    # Spec names (experiment_spec.py) use underscores; the released data uses the runtime
    # labels, which are hyphenated and say "fiction" rather than "fictional".
    subjects = {
        "same": ["peer"],
        "closed": ["closed_weight_peer", "closed-weight-peer"],
        "open": ["open_weight_peer", "open-weight-peer"],
        "fictional": ["fictional_peer", "fiction-peer"],
    }
    for paper_name in ("Gemini 3 Flash", "Gemini 3 Pro", "Claude Haiku 4.5"):
        model = MODELS[paper_name]
        t5[paper_name] = {}
        for rel, cond in RELATIONS.items():
            t5[paper_name][rel] = {}
            for col, subj in subjects.items():
                sel = mem[
                    (mem["agent_model"] == model)
                    & (mem["condition"] == cond)
                    & (mem["subject"].isin(subj))
                    & (mem["agent_harness"] == "custom")
                ]
                t5[paper_name][rel][col] = (
                    None if sel.empty else round(100 * sel["exfil"].mean(), 4)
                )
                t5[paper_name][rel][f"n_{col}"] = len(sel)

    t6: dict = {}
    for paper_name, model in (
        ("Gemini 3.1 Pro", "gemini/gemini-3.1-pro-preview"),
        ("GLM 5", "vertex_ai/zai-org/glm-5-maas"),
    ):
        t6[paper_name] = {"exfiltration": {}, "refusal": {}}
        for harness_col, harness_sel in (("custom", "custom"), ("production", "production")):
            sel = mem[
                (mem["agent_model"] == model)
                & (mem["condition"] == "good_peer")
                & (mem["subject"] == "peer")
            ]
            sel = (
                sel[sel["agent_harness"] == "custom"]
                if harness_sel == "custom"
                else sel[sel["agent_harness"] != "custom"]
            )
            t6[paper_name]["exfiltration"][harness_col] = (
                None if sel.empty else round(100 * sel["exfil"].mean(), 4)
            )
            t6[paper_name]["refusal"][harness_col] = (
                None if sel.empty else round(100 * sel["refused_b"].mean(), 4)
            )
            t6[paper_name]["exfiltration"][f"n_{harness_col}"] = len(sel)
    return {"table5_exfiltration_by_peer_identity": t5, "table6_production_harness": t6}


def compare(ours: dict, paper: dict) -> pd.DataFrame:
    rows = []

    def walk(o, p, path):
        if isinstance(p, dict):
            for k, pv in p.items():
                if k.startswith("_"):
                    continue
                walk(o.get(k) if isinstance(o, dict) else None, pv, [*path, k])
        elif isinstance(p, list):
            for i, label in enumerate(("max", "avg")):
                ov = o[i] if isinstance(o, list) else None
                rows.append(_row([*path, label], ov, p[i]))
        else:
            rows.append(_row(path, o, p))

    walk(ours, paper, [])
    return pd.DataFrame(rows)


def _row(path, ours, paper):
    diff = None if ours is None else round(ours - paper, 4)
    status = (
        "missing" if ours is None else ("match" if abs(ours - paper) <= TOLERANCE else "mismatch")
    )
    return {
        "cell": "/".join(path),
        "paper": paper,
        "ours": None if ours is None else round(ours, 4),
        "diff": diff,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", choices=("per_instantiation", "after"), default="after")
    parser.add_argument("--missing-grade", choices=("zero", "exclude"), default="exclude")
    args = parser.parse_args()

    mods = _import_upstream(args.upstream)
    csv = args.upstream / "analysis" / "data" / "merged_data.csv"
    values = per_instantiation_values(mods, csv, args.missing_grade)

    ours = reduce_cells(values, args.floor)
    ours.update(extra_tables(mods, csv))
    paper = json.loads((args.out / "paper_tables.json").read_text(encoding="utf-8"))
    cmp = compare(ours, paper)

    # The defaults are the reading that reproduces the paper; they write the answer key.
    # Other readings get suffixed files so they never overwrite it.
    is_default = (args.floor, args.missing_grade) == ("after", "exclude")
    suffix = "" if is_default else f"__floor_{args.floor}__missing_grade_{args.missing_grade}"
    values.to_csv(args.out / f"per_instantiation_values{suffix}.csv", index=False)
    (args.out / f"answer_key{suffix}.json").write_text(
        json.dumps(ours, indent=1) + "\n", encoding="utf-8"
    )
    cmp.to_csv(args.out / f"comparison_vs_paper{suffix}.csv", index=False)
    print(cmp["status"].value_counts().to_string())
    bad = cmp[cmp["status"] != "match"]
    if not bad.empty:
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
