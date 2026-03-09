#!/usr/bin/env python3
"""Comparacion robusta de modelos binarios usando predicciones por muestra.

Uso:
  python scripts/compare_three_models.py \
    --model resnet18=path/a/oof_predictions.csv \
    --model densenet121=path/b/oof_predictions.csv \
    --model efficientnetb0=path/c/oof_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


LABEL_CANDIDATES = [
    "label",
    "y_true",
    "real",
    "target",
    "ground_truth",
    "true",
]
PROB_CANDIDATES = [
    "probability",
    "prob_hernia",
    "p_pred",
    "y_prob",
    "score",
    "prob",
]
PRED_CANDIDATES = ["prediction", "pred", "y_pred"]
KEY_CANDIDATES = ["patient_id", "image_path", "imagen", "id", "sample_id"]


METRIC_DIRECTIONS = {
    "auc": "higher",
    "pr_auc": "higher",
    "accuracy": "higher",
    "f1": "higher",
    "balanced_accuracy": "higher",
    "mcc": "higher",
    "sensitivity": "higher",
    "specificity": "higher",
    "log_loss": "lower",
    "brier": "lower",
    "ece": "lower",
}

RANKING_METRICS = ["log_loss", "brier", "ece", "pr_auc", "mcc", "balanced_accuracy"]


def _find_column(df: pd.DataFrame, candidates: List[str], desc: str) -> str:
    columns = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in columns:
            return columns[cand.lower()]
    raise ValueError(f"No encontre columna para {desc}. Columnas disponibles: {list(df.columns)}")


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins, right=True) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Barrido simple de umbrales para evitar dependencias adicionales.
    thresholds = np.unique(y_prob)
    if len(thresholds) == 1:
        return float(thresholds[0])
    best_thr = 0.5
    best_j = -np.inf
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bins_ece: int) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    out = {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-8, 1 - 1e-8), labels=[0, 1])),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "ece": float(_ece_score(y_true, y_prob, n_bins=n_bins_ece)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return out


def _bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
    s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
    idx = np.concatenate([s_pos, s_neg])
    rng.shuffle(idx)
    return idx


def _bootstrap_metrics(
    y_true: np.ndarray,
    probs_by_model: Dict[str, np.ndarray],
    thr_by_model: Dict[str, float],
    n_boot: int,
    seed: int,
    n_bins_ece: int,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metrics_per_model = {name: [] for name in probs_by_model}
    for _ in range(n_boot):
        idx = _bootstrap_indices(y_true, rng)
        yb = y_true[idx]
        for name, prob in probs_by_model.items():
            pb = prob[idx]
            m = _compute_metrics(yb, pb, thr_by_model[name], n_bins_ece)
            metrics_per_model[name].append(m)
    return {name: pd.DataFrame(rows) for name, rows in metrics_per_model.items()}


def _pairwise_bootstrap(
    metrics_boot: Dict[str, pd.DataFrame], metric: str, model_a: str, model_b: str
) -> Dict[str, float]:
    a = metrics_boot[model_a][metric].to_numpy()
    b = metrics_boot[model_b][metric].to_numpy()
    if METRIC_DIRECTIONS[metric] == "higher":
        diff = a - b
    else:
        diff = b - a
    p_left = np.mean(diff <= 0)
    p_right = np.mean(diff >= 0)
    p_value = 2 * min(p_left, p_right)
    return {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "diff_oriented_mean": float(np.mean(diff)),
        "ci_low": float(np.quantile(diff, 0.025)),
        "ci_high": float(np.quantile(diff, 0.975)),
        "p_value_bootstrap": float(min(1.0, p_value)),
        "prob_a_better": float(np.mean(diff > 0)),
    }


def _mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value_mcnemar_exact": 1.0}
    p = float(binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "p_value_mcnemar_exact": p}


def _load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = _find_column(df, LABEL_CANDIDATES, "label real")
    prob_col = _find_column(df, PROB_CANDIDATES, "probabilidad")
    key_col = None
    for c in KEY_CANDIDATES:
        if c in df.columns:
            key_col = c
            break
    if key_col is None:
        key_col = "__row_id__"
        df[key_col] = np.arange(len(df))
    out = df[[key_col, label_col, prob_col]].copy()
    out.columns = ["key", "label", "probability"]
    out["label"] = out["label"].astype(int)
    out["probability"] = out["probability"].astype(float)
    return out


def _build_rank_table(metrics_point: pd.DataFrame) -> pd.DataFrame:
    rank_df = metrics_point[["model"]].copy()
    for metric in RANKING_METRICS:
        asc = METRIC_DIRECTIONS[metric] == "lower"
        rank_df[f"rank_{metric}"] = metrics_point[metric].rank(method="average", ascending=asc)
    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    rank_df["robust_rank_mean"] = rank_df[rank_cols].mean(axis=1)
    rank_df = rank_df.sort_values("robust_rank_mean").reset_index(drop=True)
    rank_df["final_position"] = np.arange(1, len(rank_df) + 1)
    return rank_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara modelos binarios de forma robusta.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Formato NOMBRE=RUTA_CSV. Repetir para cada modelo.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["youden", "fixed"],
        default="youden",
        help="Como definir umbral para metricas de clasificacion.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins-ece", type=int, default=10)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/model_comparison"),
    )
    parser.add_argument(
        "--mlflow-enable",
        action="store_true",
        help="Si se activa, registra la comparacion en MLflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Tracking URI de MLflow (ej: sqlite:///... o http://...).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="hiatal_model_comparison",
        help="Nombre del experimento MLflow para comparaciones.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Nombre del run en MLflow. Si no se define, se genera automaticamente.",
    )
    return parser.parse_args()


def _sanitize_metric_key(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)


def _log_to_mlflow(
    args: argparse.Namespace,
    parsed_models: List[Tuple[str, Path]],
    metrics_point: pd.DataFrame,
    ci_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError("No se pudo importar mlflow. Instala mlflow o ejecuta sin --mlflow-enable.") from exc

    setup_fallback_note = None
    try:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
    except Exception as exc:
        fallback_dir = (args.outdir / "mlruns_compare").resolve()
        fallback_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(fallback_dir))
        mlflow.set_experiment(args.mlflow_experiment)
        setup_fallback_note = f"Fallback a file-store MLflow por error en tracking URI principal: {exc}"

    run_name = args.mlflow_run_name or f"model_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("threshold_mode", args.threshold_mode)
        mlflow.log_param("fixed_threshold", args.fixed_threshold)
        mlflow.log_param("n_bootstrap", args.n_bootstrap)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("n_bins_ece", args.n_bins_ece)
        mlflow.log_param("n_models", len(parsed_models))
        mlflow.log_param("n_samples_compared", summary["n_samples_compared"])
        for i, (name, path) in enumerate(parsed_models, start=1):
            mlflow.log_param(f"model_{i}_name", name)
            mlflow.log_param(f"model_{i}_path", str(path))

        # Metricas puntuales por modelo.
        point_metric_cols = [c for c in metrics_point.columns if c not in {"model", "tp", "tn", "fp", "fn"}]
        for _, row in metrics_point.iterrows():
            model_name = _sanitize_metric_key(str(row["model"]))
            for col in point_metric_cols:
                val = float(row[col])
                mlflow.log_metric(f"model.{model_name}.{col}", val)

        # CIs bootstrap por modelo y metrica.
        for _, row in ci_df.iterrows():
            model_name = _sanitize_metric_key(str(row["model"]))
            metric = _sanitize_metric_key(str(row["metric"]))
            mlflow.log_metric(f"ci95.{model_name}.{metric}.low", float(row["ci_low_95"]))
            mlflow.log_metric(f"ci95.{model_name}.{metric}.high", float(row["ci_high_95"]))
            mlflow.log_metric(f"boot.{model_name}.{metric}.mean", float(row["mean_boot"]))
            mlflow.log_metric(f"boot.{model_name}.{metric}.std", float(row["std_boot"]))

        # Pruebas pareadas (bootstrap diff + McNemar).
        pair_boot = pair_df[pair_df["test"] == "bootstrap_diff"]
        for _, row in pair_boot.iterrows():
            a = _sanitize_metric_key(str(row["model_a"]))
            b = _sanitize_metric_key(str(row["model_b"]))
            m = _sanitize_metric_key(str(row["metric"]))
            base = f"pair.{a}_vs_{b}.{m}"
            mlflow.log_metric(f"{base}.diff_oriented_mean", float(row["diff_oriented_mean"]))
            mlflow.log_metric(f"{base}.ci_low", float(row["ci_low"]))
            mlflow.log_metric(f"{base}.ci_high", float(row["ci_high"]))
            mlflow.log_metric(f"{base}.p_value", float(row["p_value_bootstrap"]))
            mlflow.log_metric(f"{base}.prob_a_better", float(row["prob_a_better"]))

        pair_mc = pair_df[pair_df["test"] == "mcnemar_exact"]
        for _, row in pair_mc.iterrows():
            a = _sanitize_metric_key(str(row["model_a"]))
            b = _sanitize_metric_key(str(row["model_b"]))
            base = f"mcnemar.{a}_vs_{b}"
            mlflow.log_metric(f"{base}.b", float(row["b"]))
            mlflow.log_metric(f"{base}.c", float(row["c"]))
            mlflow.log_metric(f"{base}.p_value", float(row["p_value_mcnemar_exact"]))

        mlflow.set_tag("winner_by_rank", str(summary["winner_by_rank"]))
        mlflow.set_tag("winner_status", str(summary["winner_status"]))
        mlflow.set_tag("models", ",".join(summary["models"]))
        if setup_fallback_note:
            mlflow.set_tag("tracking_fallback_note", setup_fallback_note)

        # Artefactos.
        mlflow.log_artifact(str(args.outdir / "metrics_point_estimates.csv"), artifact_path="comparison")
        mlflow.log_artifact(str(args.outdir / "metrics_bootstrap_ci.csv"), artifact_path="comparison")
        mlflow.log_artifact(str(args.outdir / "pairwise_tests.csv"), artifact_path="comparison")
        mlflow.log_artifact(str(args.outdir / "ranking_robusto.csv"), artifact_path="comparison")
        mlflow.log_artifact(str(args.outdir / "summary.json"), artifact_path="comparison")
        mlflow.log_artifact(str(args.outdir / "report.md"), artifact_path="comparison")


def main() -> None:
    args = parse_args()
    parsed: List[Tuple[str, Path]] = []
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"--model invalido '{item}'. Use NOMBRE=RUTA")
        name, p = item.split("=", 1)
        parsed.append((name.strip(), Path(p).expanduser()))
    if len(parsed) < 3:
        raise ValueError("Se requieren al menos 3 modelos para comparar.")

    dfs = {}
    for name, path in parsed:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")
        dfs[name] = _load_predictions(path)

    # Alineacion por llave compartida.
    base_name = parsed[0][0]
    base = dfs[base_name][["key", "label"]].copy()
    merged = base
    for name, _ in parsed:
        cur = dfs[name][["key", "label", "probability"]].copy()
        cur = cur.rename(columns={"label": f"label_{name}", "probability": f"prob_{name}"})
        merged = merged.merge(cur, on="key", how="inner")
    if len(merged) == 0:
        raise ValueError("No hubo interseccion de muestras entre los modelos.")

    # Validar consistencia de labels entre modelos.
    for name, _ in parsed:
        if not np.array_equal(merged["label"].to_numpy(), merged[f"label_{name}"].to_numpy()):
            raise ValueError(f"Labels inconsistentes para el modelo {name}.")

    y_true = merged["label"].to_numpy(dtype=int)
    probs_by_model = {name: merged[f"prob_{name}"].to_numpy(dtype=float) for name, _ in parsed}

    if args.threshold_mode == "youden":
        thr_by_model = {n: _youden_threshold(y_true, p) for n, p in probs_by_model.items()}
    else:
        thr_by_model = {n: float(args.fixed_threshold) for n in probs_by_model}

    metrics_point_rows = []
    preds_by_model = {}
    for name, _ in parsed:
        m = _compute_metrics(y_true, probs_by_model[name], thr_by_model[name], args.n_bins_ece)
        preds_by_model[name] = (probs_by_model[name] >= thr_by_model[name]).astype(int)
        m["model"] = name
        metrics_point_rows.append(m)
    metrics_point = pd.DataFrame(metrics_point_rows).sort_values("model").reset_index(drop=True)

    metrics_boot = _bootstrap_metrics(
        y_true=y_true,
        probs_by_model=probs_by_model,
        thr_by_model=thr_by_model,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        n_bins_ece=args.n_bins_ece,
    )

    ci_rows = []
    metric_names = [m for m in METRIC_DIRECTIONS.keys()]
    for name in probs_by_model:
        dfm = metrics_boot[name]
        for metric in metric_names:
            vals = dfm[metric].to_numpy()
            ci_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean_boot": float(np.mean(vals)),
                    "std_boot": float(np.std(vals, ddof=1)),
                    "ci_low_95": float(np.quantile(vals, 0.025)),
                    "ci_high_95": float(np.quantile(vals, 0.975)),
                }
            )
    ci_df = pd.DataFrame(ci_rows)

    pair_rows = []
    names = [n for n, _ in parsed]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for metric in ["log_loss", "brier", "ece", "pr_auc", "mcc", "balanced_accuracy", "auc"]:
                row = _pairwise_bootstrap(metrics_boot, metric, a, b)
                row["test"] = "bootstrap_diff"
                pair_rows.append(row)
            mc = _mcnemar(y_true, preds_by_model[a], preds_by_model[b])
            pair_rows.append(
                {
                    "metric": "accuracy_error_pattern",
                    "model_a": a,
                    "model_b": b,
                    "test": "mcnemar_exact",
                    "diff_oriented_mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value_bootstrap": np.nan,
                    "prob_a_better": np.nan,
                    **mc,
                }
            )
    pair_df = pd.DataFrame(pair_rows)

    ranking_df = _build_rank_table(metrics_point)
    winner = ranking_df.iloc[0]["model"]

    # Verificar si el ganador tiene ventaja clara en log_loss y brier.
    winner_checks = pair_df[
        (pair_df["test"] == "bootstrap_diff")
        & (pair_df["metric"].isin(["log_loss", "brier"]))
        & (
            ((pair_df["model_a"] == winner) & (pair_df["ci_low"] > 0))
            | ((pair_df["model_b"] == winner) & (pair_df["ci_high"] < 0))
        )
    ]
    n_expected = 2 * (len(names) - 1)
    tie_msg = (
        "Ganador estadistico claro."
        if len(winner_checks) >= n_expected
        else "No hay ganador estadisticamente concluyente (empate tecnico parcial)."
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_point.to_csv(args.outdir / "metrics_point_estimates.csv", index=False)
    ci_df.to_csv(args.outdir / "metrics_bootstrap_ci.csv", index=False)
    pair_df.to_csv(args.outdir / "pairwise_tests.csv", index=False)
    ranking_df.to_csv(args.outdir / "ranking_robusto.csv", index=False)

    report = {
        "n_samples_compared": int(len(y_true)),
        "models": names,
        "threshold_mode": args.threshold_mode,
        "thresholds": thr_by_model,
        "winner_by_rank": winner,
        "winner_status": tie_msg,
        "ranking_metrics": RANKING_METRICS,
    }
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Comparación robusta de modelos",
        "",
        f"- Muestras comparadas: **{len(y_true)}**",
        f"- Modelos: **{', '.join(names)}**",
        f"- Modo de umbral: **{args.threshold_mode}**",
        f"- Mejor por ranking robusto: **{winner}**",
        f"- Estado: **{tie_msg}**",
        "",
        "## Archivos generados",
        "- `metrics_point_estimates.csv`",
        "- `metrics_bootstrap_ci.csv`",
        "- `pairwise_tests.csv`",
        "- `ranking_robusto.csv`",
        "- `summary.json`",
    ]
    (args.outdir / "report.md").write_text("\n".join(md), encoding="utf-8")

    if args.mlflow_enable:
        _log_to_mlflow(
            args=args,
            parsed_models=parsed,
            metrics_point=metrics_point,
            ci_df=ci_df,
            pair_df=pair_df,
            ranking_df=ranking_df,
            summary=report,
        )

    print("\n".join(md))


if __name__ == "__main__":
    main()
