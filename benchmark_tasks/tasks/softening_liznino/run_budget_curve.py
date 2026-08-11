from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENV_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ENV_DIR / "results"
CSV_FIELDS = [
    "env_id",
    "baseline",
    "model",
    "seed",
    "budget",
    "round",
    "energy_mae_per_atom",
    "force_mae",
    "stress_mae",
    "missing_predictions",
    "hard_gate_violations",
    "train_minutes",
    "gpu_minutes",
    "notes",
]


def score_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if "metrics" not in data or data.get("split") != "hidden_test":
            continue
        metrics = data["metrics"]
        rows.append(
            {
                "env_id": data.get("env_id"),
                "baseline": data.get("baseline"),
                "model": data.get("model"),
                "seed": data.get("seed"),
                "budget": data.get("budget"),
                "round": data.get("round"),
                "energy_mae_per_atom": metrics.get("energy_mae_per_atom"),
                "force_mae": metrics.get("force_mae"),
                "stress_mae": metrics.get("stress_mae"),
                "missing_predictions": metrics.get("missing_predictions"),
                "hard_gate_violations": ";".join(metrics.get("hard_gate_violations", [])),
                "train_minutes": data.get("train_minutes", ""),
                "gpu_minutes": data.get("gpu_minutes", ""),
                "notes": data.get("notes", ""),
            }
        )
    return sorted(rows, key=lambda r: (str(r["baseline"]), int(r["budget"]), int(r["seed"] or 0)))


def write_curve(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "score_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stats_by_baseline(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["baseline"]), int(row["budget"]))].append(row)
    stats = {}
    for (baseline, budget), items in grouped.items():
        force = np.asarray([float(i["force_mae"]) for i in items if i["force_mae"] not in (None, "")], dtype=float)
        energy = np.asarray([float(i["energy_mae_per_atom"]) for i in items if i["energy_mae_per_atom"] not in (None, "")], dtype=float)
        stats.setdefault(baseline, []).append(
            {
                "budget": budget,
                "n": len(items),
                "force_mae_mean": float(np.mean(force)) if len(force) else None,
                "force_mae_std": float(np.std(force, ddof=0)) if len(force) else None,
                "energy_mae_per_atom_mean": float(np.mean(energy)) if len(energy) else None,
                "energy_mae_per_atom_std": float(np.std(energy, ddof=0)) if len(energy) else None,
            }
        )
    for baseline in stats:
        stats[baseline] = sorted(stats[baseline], key=lambda x: x["budget"])
    return stats


def write_summary(rows: list[dict]) -> None:
    budgets = sorted({int(row["budget"]) for row in rows if row["budget"] is not None})
    stats = stats_by_baseline(rows)
    endpoint = None
    if "random_oracle_naive_finetune" in stats:
        endpoint = stats["random_oracle_naive_finetune"][-1]
    summary = {
        "env_id": rows[0]["env_id"] if rows else "BENCHMARK-SOFT-LIZNINIO-004",
        "score_vs_budget": {
            "budgets": budgets,
            "metrics_by_budget": stats,
        },
        "endpoint_score": endpoint,
        "oracle_cost": {"max_n_frames": max(budgets) if budgets else 0},
        "gpu_cost": {
            "train_wall_time_s": None,
            "eval_wall_time_s": None,
            "gpu_hours": None,
            "note": "Per-run wall times are in logs; summary extraction is pending.",
        },
        "hard_gate_violations": sorted({v for row in rows for v in str(row.get("hard_gate_violations", "")).split(";") if v}),
        "downstream_observable_error": None,
        "missing_predictions": {
            "max_missing_predictions": max([int(row["missing_predictions"] or 0) for row in rows], default=0)
        },
        "decision_trace_summary": {
            "status": "first_minimal_curve_complete",
            "note": "Reduced WBM softening environment; not the LiZnInIO low-data trajectory case.",
        },
    }
    (RESULTS_DIR / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def write_plot(rows: list[dict]) -> None:
    stats = stats_by_baseline(rows)
    plt.figure(figsize=(6.4, 4.2))
    for baseline, items in stats.items():
        x = np.asarray([i["budget"] for i in items], dtype=float)
        y = np.asarray([i["force_mae_mean"] for i in items], dtype=float)
        yerr = np.asarray([i["force_mae_std"] or 0.0 for i in items], dtype=float)
        label = "zero-shot" if baseline == "zero_shot" else "random oracle + naive FT"
        plt.plot(x, y, marker="o", label=label)
        if len(items) > 1 or np.any(yerr > 0):
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.18)
    plt.xlabel("oracle budget (# labeled frames)")
    plt.ylabel("hidden force MAE (eV/A)")
    plt.title("BENCHMARK-SOFT-LIZNINIO-004 reduced WBM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "score_vs_budget.png", dpi=180)
    plt.close()


def main() -> None:
    rows = score_rows()
    write_curve(rows)
    write_summary(rows)
    write_plot(rows)
    print(f"wrote {RESULTS_DIR / 'score_curve.csv'} ({len(rows)} rows)")
    print(f"wrote {RESULTS_DIR / 'score_summary.json'}")
    print(f"wrote {RESULTS_DIR / 'score_vs_budget.png'}")


if __name__ == "__main__":
    main()
