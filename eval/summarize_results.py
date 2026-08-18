
import json
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "eval_results.json"
METRICS = ["f1", "rougeL", "semantic"]


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def summarize(rows):
    summary = {}
    for metric in METRICS:
        values = [r[metric] for r in rows]
        q1, median, q3 = st.quantiles(values, n=4, method="inclusive")
        summary[metric] = {
            "mean": st.mean(values),
            "min": min(values),
            "max": max(values),
            "stdev": st.stdev(values) if len(values) > 1 else 0.0,
            "q1": q1,
            "median": median,
            "q3": q3,
        }
    return summary


def print_summary(rows, summary):
    print(f"n = {len(rows)} preguntas\n")
    print(
        f"{'Metrica':<10} {'Promedio':>9} {'Minimo':>9} {'Q1':>9} {'Mediana':>9} "
        f"{'Q3':>9} {'Maximo':>9} {'Desv.Std':>9}"
    )
    print("-" * 90)
    for metric in METRICS:
        s = summary[metric]
        print(
            f"{metric:<10} {s['mean']:>9.3f} {s['min']:>9.3f} {s['q1']:>9.3f} "
            f"{s['median']:>9.3f} {s['q3']:>9.3f} {s['max']:>9.3f} {s['stdev']:>9.3f}"
        )


def main():
    rows = load_results()
    summary = summarize(rows)
    print_summary(rows, summary)


if __name__ == "__main__":
    main()
